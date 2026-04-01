import torch
if __name__ == "__main__":
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

import os
import json
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import math
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def is_numeric_label(label):
    if re.search(r'\d', label): return True
    ordinals = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덞째", "하옵째", "열째"]
    if any(label.startswith(o) for o in ordinals): return True
    numeric_suffixes = ["회", "시간", "분", "시", "일", "월", "달", "년", "km", "명", "살", "호선"]
    if any(label.endswith(s) for s in numeric_suffixes): return True
    return False

def calculate_hand_angles(hand):
    T = hand.size(0)
    def get_angle(v1, v2):
        dot = (v1 * v2).sum(dim=-1)
        norm1 = torch.norm(v1, dim=-1) + 1e-9
        norm2 = torch.norm(v2, dim=-1) + 1e-9
        return torch.acos(torch.clamp(dot / (norm1 * norm2), -1.0, 1.0))
    v01 = hand[:, 1] - hand[:, 0]; v12 = hand[:, 2] - hand[:, 1]; a_th_cmc = get_angle(v01, v12)
    v23 = hand[:, 3] - hand[:, 2]; a_th_mcp = get_angle(v12, v23)
    v34 = hand[:, 4] - hand[:, 3]; a_th_ip = get_angle(v23, v34)
    v05 = hand[:, 5] - hand[:, 0]; v56 = hand[:, 6] - hand[:, 5]; a_idx_mcp = get_angle(v05, v56)
    v09 = hand[:, 9] - hand[:, 0]; v910 = hand[:, 10] - hand[:, 9]; a_mid_mcp = get_angle(v09, v910)
    v013 = hand[:, 13] - hand[:, 0]; v1314 = hand[:, 14] - hand[:, 13]; a_ring_mcp = get_angle(v013, v1314)
    v017 = hand[:, 17] - hand[:, 0]; v1718 = hand[:, 18] - hand[:, 17]; a_pnk_mcp = get_angle(v017, v1718)
    return torch.stack([a_th_cmc, a_th_mcp, a_th_ip, a_idx_mcp, a_mid_mcp, a_ring_mcp, a_pnk_mcp], dim=1)

def apply_canonical_hand_normalization(hand_tensor):
    T = hand_tensor.size(0)
    hand = hand_tensor.view(T, 21, 3)
    angles = calculate_hand_angles(hand)
    wrist = hand[:, 0:1, :]
    hand = hand - wrist
    middle_mcp = hand[:, 9, :]
    hand_scale = torch.norm(middle_mcp, dim=1, keepdim=True).unsqueeze(-1) + 1e-9
    hand = hand / hand_scale
    tips_idx = [4, 8, 12, 16, 20]
    tips = hand[:, tips_idx, :]
    distances = torch.norm(tips, dim=2)
    return hand.view(T, 63), distances, angles, wrist.view(T, 3), hand_scale.view(T, 1)

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.reduction, self.smoothing = alpha, gamma, reduction, smoothing
    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        f_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            f_weight *= self.alpha.gather(0, targets).unsqueeze(1)
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs).fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        target_f_weight = f_weight.gather(1, targets.unsqueeze(1)).squeeze()
        loss = torch.sum(-true_dist * log_pt, dim=1) * target_f_weight
        if self.reduction == 'mean': return loss.mean()
        return loss.sum()

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, float('inf')
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(val_loss, model); self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path); self.val_loss_min = val_loss

def load_datasets(processed_dir='processed/', top_n_classes=1500):
    def load_shards(split):
        shard_dir = os.path.join(processed_dir, split)
        if not os.path.exists(shard_dir): raise FileNotFoundError(f"Missing: {shard_dir}")
        all_p, all_l, all_r, all_f, all_lbl = [], [], [], [], []
        files = sorted([f for f in os.listdir(shard_dir) if f.endswith('.pt')])
        for f in files:
            s = torch.load(os.path.join(shard_dir, f))
            all_p.append(s['pose']); all_l.append(s['left_hand']); all_r.append(s['right_hand']); all_f.append(s['face'])
            all_lbl.extend([lbl.strip().split('\n')[0] for lbl in s['labels']])
        X = {'pose': torch.cat(all_p), 'left_hand': torch.cat(all_l), 'right_hand': torch.cat(all_r), 'face': torch.cat(all_f)}
        return X, all_lbl
    (X_tr, y_tr_s), (X_vl, y_vl_s), (X_ts, y_ts_s) = load_shards('train'), load_shards('val'), load_shards('test')
    common = set(y_tr_s) & set(y_vl_s) & set(y_ts_s)
    
    # 지숫자만 필터링 (is_numeric_label == True)
    num_labels = [lbl for lbl in y_tr_s if is_numeric_label(lbl)]
    top_labels = {lbl: i for i, (lbl, _) in enumerate(Counter(num_labels).most_common()) if lbl in common}
    top_labels = {lbl: i for i, lbl in enumerate(list(top_labels.keys())[:top_n_classes])}
    
    def filter_data(X, y_s):
        mt = torch.tensor([top_labels.get(y, -1) for y in y_s], dtype=torch.long)
        mask = mt != -1
        return {k: v[mask] for k, v in X.items()}, mt[mask]
    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts) = filter_data(X_tr, y_tr_s), filter_data(X_vl, y_vl_s), filter_data(X_ts, y_ts_s)
    return (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts), top_labels

class SignDataset(Dataset):
    def __init__(self, X, y, face_weight=0.0):
        self.X, self.y = X, y
        self.face_weight = face_weight
        
    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        lh_raw = self.X['left_hand'][idx]
        rh_raw = self.X['right_hand'][idx]
        
        lh_norm, lh_dist, lh_angle, lh_wrist, lh_scale = apply_canonical_hand_normalization(lh_raw)
        rh_norm, rh_dist, rh_angle, rh_wrist, rh_scale = apply_canonical_hand_normalization(rh_raw)
        
        # 손목 궤적 Zero-centering (절대 위치 삭제) 후 손 크기로 나누어 스케일 외우기(컨닝) 방지!
        lh_wrist = (lh_wrist - lh_wrist.mean(dim=0, keepdim=True)) / lh_scale
        rh_wrist = (rh_wrist - rh_wrist.mean(dim=0, keepdim=True)) / rh_scale
        
        # 양손 각각 78차원(63+5+7+3) = 156차원
        feat = torch.cat([lh_norm, lh_dist, lh_angle, lh_wrist, rh_norm, rh_dist, rh_angle, rh_wrist], dim=1)
        # 각 프레임의 절대적 위치를 지우기 위해 시간 축(dim=0)에 대한 평균만 빼기 (Zero-centering)
        feat = feat - feat.mean(dim=0, keepdim=True)
        # 중요! 각 관절별(dim=0) 개별 std로 나누게 되면 움직이지 않는 관절(어깨나 골반)의 미세한 떨림이 1만배 뻥튀기 되어 정답을 외우는 커닝페이퍼(Overfitting)가 됩니다!
        # 따라서 전체의 글로벌 분산(글로벌 std) 하나로만 스케일링하여 움직임이 큰 손은 크게, 안 움직인 뼈는 0에 수렴하게 유지해야 합니다.
        feat = feat / (feat.std() + 1e-8)
        
        return feat.float(), int(self.y[idx])

class PreprocessedSignDataset(Dataset):
    def __init__(self, dataset, split='train', augment=True):
        self.dataset = dataset; self.split = split; self.augment = augment and split == 'train'
        self.aug_factor = 1 # 숫자 데이터셋 단독 학습이므로 과적합 방지를 위해 1배수로 롤백

    def __len__(self): return len(self.dataset) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % len(self.dataset)
        feat, label = self.dataset[orig_idx]
        if self.augment: 
            # 매 에포크마다 0~9번 증강 중 하나를 랜덤하게 적용하여 무한한 다양성 부여
            aug_idx = random.randint(0, 9) 
            feat = self._apply_augmentation(feat, aug_idx)
        return feat, label

    def _apply_augmentation(self, f, aug_idx):
        if aug_idx == 0: return f
        if aug_idx in [1, 5, 8]: 
            scale = random.uniform(0.8, 1.2)
            l = max(1, int(len(f) * scale))
            res = F.interpolate(f.transpose(0,1).unsqueeze(0), size=l, mode='linear', align_corners=True).squeeze(0).transpose(0,1)
            if l < 150: res = F.pad(res, (0,0,0,150-l))
            return res[:150]
        if aug_idx in [2, 6, 9]:
            # 시간축 무작위 자르기(Temporal Crop/Dropout) 기법 적용
            res = f.clone()
            drop_len = random.randint(5, 20)
            start_idx = random.randint(0, max(0, len(f) - drop_len))
            res[start_idx:start_idx+drop_len, :] = 0.0
            return res
        if aug_idx == 3:
            mask = torch.rand(f.size(0)) > 0.2
            if mask.any(): 
                res = f[mask]
                return F.interpolate(res.transpose(0,1).unsqueeze(0), size=150, mode='linear', align_corners=True).squeeze(0).transpose(0,1)
            return f
        if aug_idx in [4, 7]:
            return f + torch.randn_like(f) * 0.02 # 노이즈를 줄여 숫자 수어의 미세한 손가락 각도 보존
        return f

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class HandSelfAttention(nn.Module):
    def __init__(self, hand_dim=128, embed_dim=128, num_heads=4):
        super().__init__()
        self.hand_proj = nn.Linear(hand_dim, embed_dim)
        self.attn_hand = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.5)
        self.norm_hand = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(embed_dim, 128) # 모델 크기 극한 단축 (256 -> 128)

    def forward(self, h_f):
        # h_f: (B, C, T)
        h = h_f.permute(2, 0, 1) # (T, B, C)
        h_emb = self.hand_proj(h)
        a_h, _ = self.attn_hand(h_emb, h_emb, h_emb)
        h_res = self.norm_hand(h_emb + self.dropout(a_h))
        fused = self.out_proj(h_res)
        return fused.permute(1, 2, 0), h_res.permute(1, 2, 0)

class ModalityFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lh_enc = nn.Sequential(nn.Conv1d(78, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.rh_enc = nn.Sequential(nn.Conv1d(78, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.self_attn = HandSelfAttention(hand_dim=128, embed_dim=128)

    def forward(self, x):
        lh = x[:,:,0:78]
        rh = x[:,:,78:156]
        lh_f = self.lh_enc(lh.transpose(1,2))
        rh_f = self.rh_enc(rh.transpose(1,2))
        h_f = torch.cat([lh_f, rh_f], dim=1)
        return self.self_attn(h_f)

class SuffixAwareAttentionPooling(nn.Module):
    def __init__(self, channels, seq_len=150):
        super().__init__()
        self.attn_proj = nn.Linear(channels, 1)
        suffix_bias = torch.linspace(-1.0, 1.0, seq_len).view(1, seq_len, 1)
        self.suffix_bias = nn.Parameter(suffix_bias)
        
    def forward(self, x):
        T = x.size(2)
        scores = self.attn_proj(x.transpose(1, 2))
        if T == self.suffix_bias.size(1): scores = scores + self.suffix_bias
        else: scores = scores + F.interpolate(self.suffix_bias.permute(0, 2, 1), size=T, mode='linear', align_corners=True).permute(0, 2, 1)
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights.transpose(1, 2)).sum(dim=2)
        return pooled

class TCNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fusion = ModalityFusionModule()
        self.tcn = nn.ModuleList([nn.Conv1d(128, 128, 3, padding=d, dilation=d) for d in [1, 2, 4]])
        self.bn = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(3)])
        self.droppath = nn.ModuleList([DropPath(0.3) for _ in range(3)])
        self.dropout_tcn = nn.Dropout(0.6)
        
        self.suffix_attn = SuffixAwareAttentionPooling(128)
        
        self.fc1 = nn.Linear(128, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.hand_classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        fused, hands = self.fusion(x)
        for conv, bn, dp in zip(self.tcn, self.bn, self.droppath):
            res = fused
            fused = F.relu(bn(conv(fused)))
            fused = self.dropout_tcn(fused)
            if fused.size(2) == res.size(2): fused = res + dp(fused)
            
        x_p = self.suffix_attn(fused)
        
        m_feat = F.relu(self.bn_fc1(self.fc1(x_p)))
        m_logits = self.fc2(F.dropout(m_feat, 0.5, self.training))
        h_logits = self.hand_classifier(hands.mean(dim=2))
        return m_logits + 1.0 * h_logits # 숫자는 손의 비중이 절대적임

def train_epoch(model, loader, criterion, optimizer, device, scaler, scheduler=None):
    model.train(); tl, cor, tot = 0, 0, 0
    for s, l in tqdm(loader, desc='Training'):
        s, l = s.to(device), l.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(s)
            loss = criterion(out, l)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
        tl += loss.item(); _, p = out.max(1); tot += l.size(0); cor += p.eq(l).sum().item()
    return tl/len(loader), 100.*cor/tot

def evaluate(model, loader, criterion, device):
    model.eval(); tl, c1, c5, tot = 0, 0, 0, 0
    with torch.no_grad():
        for s, l in loader:
            s, l = s.to(device), l.to(device)
            out = model(s)
            tl += criterion(out, l).item(); tot += l.size(0)
            c1 += out.max(1)[1].eq(l).sum().item()
            # 클래스가 5개 미만일 수도 있으니 안전하게 처리
            k = min(5, out.size(1))
            c5 += (out.topk(k, 1)[1] == l.unsqueeze(-1)).any(1).sum().item()
    return tl/len(loader), 100.*c1/tot, 100.*c5/tot

def collate_fn(batch):
    p = pad_sequence([it[0] for it in batch], batch_first=True)
    l = torch.tensor([it[1] for it in batch], dtype=torch.long)
    return p, l

if __name__ == "__main__":
    def is_colab():
        try: return 'google.colab' in str(get_ipython())
        except: return False
    BASE_DIR = '/content/drive/MyDrive/processed' if is_colab() else 'processed'
    LR = 0.0004

    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts), label_map = load_datasets(processed_dir=BASE_DIR, top_n_classes=1500)
    print(f"Num Numeric Classes: {len(label_map)}")
    
    tr_ds_raw = SignDataset(X_tr, y_tr, face_weight=0.0) # 숫자는 얼굴 중요도 없음
    tr_ds = PreprocessedSignDataset(tr_ds_raw, split='train', augment=True)
    
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    vl_loader = DataLoader(PreprocessedSignDataset(SignDataset(X_vl, y_vl, face_weight=0.0), augment=False), batch_size=128, collate_fn=collate_fn)
    ts_loader = DataLoader(PreprocessedSignDataset(SignDataset(X_ts, y_ts, face_weight=0.0), augment=False), batch_size=128, collate_fn=collate_fn)

    model = TCNClassifier(len(label_map)).to('cuda')
    # counts = torch.bincount(y_tr, minlength=len(label_map)).float()
    # cw = 1.0 / (torch.sqrt(counts) + 1e-6); cw /= cw.mean()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.08)
    epochs = 200
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=epochs)
    early_stop = EarlyStopping(patience=30, verbose=True, path='best_ver3_0_num.pt')
    scaler = torch.amp.GradScaler('cuda')

    print(f"Version 3.0 Number Classifier Training Started")
    for epoch in range(epochs):
        t_l, t_a = train_epoch(model, tr_loader, criterion, optimizer, 'cuda', scaler, scheduler)
        v_l, v_a, v_5 = evaluate(model, vl_loader, criterion, 'cuda')
        print(f"Epoch {epoch+1}/{epochs}\nTrain Loss: {t_l:.4f}, Train Acc: {t_a:.2f}%\nVal Loss: {v_l:.4f}, Val Acc: {v_a:.2f}%, Val Top-5: {v_5:.2f}%")
        early_stop(v_l, model)
        if early_stop.early_stop: break
    
    if os.path.exists('best_ver3_0_num.pt'):
        model.load_state_dict(torch.load('best_ver3_0_num.pt'))
    _, ts_a, ts_5 = evaluate(model, ts_loader, criterion, 'cuda')
    print(f"\nFinal Test Acc (V3.0 Num): {ts_a:.2f}%, Top-5: {ts_5:.2f}%")