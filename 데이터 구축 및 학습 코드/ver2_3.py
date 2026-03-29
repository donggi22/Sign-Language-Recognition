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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import random
import math
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# --- 유틸리티 및 전처리 ---

def is_numeric_label(label):
    if re.search(r'\d', label): return True
    ordinals = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덞째", "하옵째", "열째"]
    if any(label.startswith(o) for o in ordinals): return True
    numeric_suffixes = ["회", "시간", "분", "시", "일", "월", "달", "년", "km", "명", "살", "호선"]
    if any(label.endswith(s) for s in numeric_suffixes): return True
    return False

def apply_canonical_hand_normalization(hand_tensor):
    """
    hand_tensor: [T, 63] (21 landmarks x 3)
    Returns: [T, 63] normalized landmarks + [T, 5] fingertip distances
    """
    T = hand_tensor.size(0)
    hand = hand_tensor.view(T, 21, 3)
    
    # 1. Wrist-centric Normalization (Index 0 is wrist)
    wrist = hand[:, 0:1, :] # [T, 1, 3]
    hand = hand - wrist
    
    # 2. Scale Invariance (Normalize by distance between wrist(0) and middle MCP(9))
    # Avoid division by zero
    middle_mcp = hand[:, 9, :] # [T, 3]
    hand_scale = torch.norm(middle_mcp, dim=1, keepdim=True).unsqueeze(-1) + 1e-9
    hand = hand / hand_scale
    
    # 3. Selective Geometric Features: Fingertip to Wrist distances
    # Tips: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    tips_idx = [4, 8, 12, 16, 20]
    tips = hand[:, tips_idx, :] # [T, 5, 3]
    distances = torch.norm(tips, dim=2) # [T, 5] (Already scaled since 'hand' is scaled)
    
    return hand.view(T, 63), distances

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.reduction, self.smoothing = alpha, gamma, reduction, smoothing
    def forward(self, inputs, targets, per_sample=False):
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
        
        if per_sample: return loss
        if self.reduction == 'mean': return loss.mean()
        return loss.sum()

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='best_ver2_3.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, float('inf')
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(val_loss, model); self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path); self.val_loss_min = val_loss

def load_datasets(processed_dir='processed/', top_n_classes=994):
    def load_shards(split):
        shard_dir = os.path.join(processed_dir, split)
        if not os.path.exists(shard_dir): raise FileNotFoundError(f"Missing: {shard_dir}")
        all_p, all_l, all_r, all_f, all_lbl = [], [], [], [], []
        files = sorted([f for f in os.listdir(shard_dir) if f.endswith('.pt')])
        if not files: raise FileNotFoundError(f"Empty: {shard_dir}")
        for f in files:
            s = torch.load(os.path.join(shard_dir, f))
            all_p.append(s['pose']); all_l.append(s['left_hand']); all_r.append(s['right_hand']); all_f.append(s['face'])
            all_lbl.extend([lbl.strip().split('\n')[0] for lbl in s['labels']])
        X = {'pose': torch.cat(all_p), 'left_hand': torch.cat(all_l), 'right_hand': torch.cat(all_r), 'face': torch.cat(all_f)}
        return X, all_lbl
    (X_tr, y_tr_s), (X_vl, y_vl_s), (X_ts, y_ts_s) = load_shards('train'), load_shards('val'), load_shards('test')
    common = set(y_tr_s) & set(y_vl_s) & set(y_ts_s)
    top_labels = {lbl: i for i, (lbl, _) in enumerate(Counter(y_tr_s).most_common()) if lbl in common}
    top_labels = {lbl: i for i, lbl in enumerate(list(top_labels.keys())[:top_n_classes])}
    def filter_data(X, y_s):
        mt = torch.tensor([top_labels.get(y, -1) for y in y_s], dtype=torch.long)
        mask = mt != -1
        return {k: v[mask] for k, v in X.items()}, mt[mask]
    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts) = filter_data(X_tr, y_tr_s), filter_data(X_vl, y_vl_s), filter_data(X_ts, y_ts_s)
    return (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts), top_labels

class SignDataset(Dataset):
    def __init__(self, X, y, label_map, face_weight=2.0):
        self.X, self.y, self.label_map = X, y, label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.face_weight = face_weight
        self.is_numeric = [is_numeric_label(self.idx_to_label[int(ly)]) for ly in y]
        
    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        is_num = self.is_numeric[idx]
        current_face_weight = self.face_weight * 0.5 if is_num else self.face_weight
        
        p = self.X['pose'][idx]
        lh_raw = self.X['left_hand'][idx]
        rh_raw = self.X['right_hand'][idx]
        f = self.X['face'][idx]
        
        # Robust Normalization & Feature Extraction
        lh_norm, lh_dist = apply_canonical_hand_normalization(lh_raw)
        rh_norm, rh_dist = apply_canonical_hand_normalization(rh_raw)
        
        # Concatenate Hand + Pose + Face + Geo Features
        # LH (63) + LH_dist(5) + RH(63) + RH_dist(5) + Pose(75) + Face(210) = 421
        feat = torch.cat([p, lh_norm, lh_dist, rh_norm, rh_dist, f * current_face_weight], dim=1)
        
        # Double Normalization
        feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-8)
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        
        return feat.float(), int(self.y[idx]), int(is_num)

class PreprocessedSignDataset(Dataset):
    def __init__(self, dataset, split='train', augment=True):
        self.dataset = dataset
        self.split = split
        self.augment = augment and split == 'train'
        self.aug_factor = 6 if self.augment else 1

    def __len__(self): return len(self.dataset) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % len(self.dataset)
        aug_idx  = idx // len(self.dataset)
        feat, label, is_num = self.dataset[orig_idx]
        if self.augment:
            feat = self._apply_augmentation(feat, aug_idx, is_num)
        return feat, label, is_num

    def _apply_augmentation(self, f, aug_idx, is_num):
        if aug_idx == 0: return f
        
        if aug_idx == 1: # Time Scaling
            scale = random.uniform(0.8, 1.2)
            l = max(1, int(len(f) * scale))
            res = F.interpolate(f.transpose(0,1).unsqueeze(0), size=l, mode='linear', align_corners=True).squeeze(0).transpose(0,1)
            if l < 150: res = F.pad(res, (0,0,0,150-l))
            return res[:150]
            
        if aug_idx == 2: # Landmark Jittering (Specially important for numeric)
            noise_std = 0.02 if is_num else 0.015
            return f + torch.randn_like(f) * noise_std
            
        if aug_idx == 3: # Frame Dropout
            p = 0.2 if is_num else 0.1
            mask = torch.rand(f.size(0)) > p
            if mask.any(): 
                res = f[mask]
                return F.interpolate(res.transpose(0,1).unsqueeze(0), size=150, mode='linear', align_corners=True).squeeze(0).transpose(0,1)
            return f

        if aug_idx == 4: # Random Shift Jitter
            shift = torch.randn(1, f.size(1)) * 0.01
            return f + shift

        if aug_idx == 5: # Random Scale Jitter
            scale = random.uniform(0.95, 1.05)
            return f * scale
            
        return f

# --- 모델 구조 ---

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

class CrossModalAttention(nn.Module):
    def __init__(self, pose_dim, hand_dim, face_dim, embed_dim=128, num_heads=4):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, embed_dim)
        self.hand_proj = nn.Linear(hand_dim, embed_dim)
        self.face_proj = nn.Linear(face_dim, embed_dim)
        self.attn_pose = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2)
        self.attn_hand = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2)
        self.norm_pose = nn.LayerNorm(embed_dim)
        self.norm_hand = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(pose_dim + hand_dim + face_dim + embed_dim * 2, 384)

    def forward(self, p_f, lh_f, rh_f, f_f, is_numeric=None):
        p = p_f.permute(2, 0, 1) 
        h = torch.cat([lh_f, rh_f], dim=1).permute(2, 0, 1) 
        f = f_f.permute(2, 0, 1) 
        
        p_emb, h_emb, f_emb = self.pose_proj(p), self.hand_proj(h), self.face_proj(f)
        
        ctx_p = torch.cat([h_emb, f_emb], 0)
        a_p, _ = self.attn_pose(p_emb, ctx_p, ctx_p)
        p_res = self.norm_pose(p_emb + self.dropout(a_p))
        
        ctx_h = torch.cat([p_emb, h_emb], 0)
        a_h, _ = self.attn_hand(h_emb, ctx_h, ctx_h)
        h_res = self.norm_hand(h_emb + self.dropout(a_h))
        
        if is_numeric is not None:
            scale = 1.0 + is_numeric.unsqueeze(0).unsqueeze(-1).float() * 0.5
            h_res = h_res * scale

        fused = torch.cat([p.permute(1,0,2), h.permute(1,0,2), f.permute(1,0,2), 
                          p_res.permute(1,0,2), h_res.permute(1,0,2)], 2)
        fused = self.out_proj(fused)
        return fused.permute(0, 2, 1), h_res.permute(1, 2, 0)

class ModalityFusionModule(nn.Module):
    def __init__(self, pose_dim=75, hand_dim=136, face_dim=210): 
        super().__init__()
        self.p_enc = nn.Sequential(nn.Conv1d(pose_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.lh_enc = nn.Sequential(nn.Conv1d(68, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rh_enc = nn.Sequential(nn.Conv1d(68, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.f_enc = nn.Sequential(nn.Conv1d(face_dim, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU())
        self.cross_attn = CrossModalAttention(128, 256, 256)

    def forward(self, x, is_numeric=None):
        p = x[:,:,:75]
        lh = x[:,:,75:143] 
        rh = x[:,:,143:211] 
        f = x[:,:,211:]
        
        p_f = self.p_enc(p.transpose(1,2))
        lh_f = self.lh_enc(lh.transpose(1,2))
        rh_f = self.rh_enc(rh.transpose(1,2))
        f_f = self.f_enc(f.transpose(1,2))
        return self.cross_attn(p_f, lh_f, rh_f, f_f, is_numeric)

class TCNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fusion = ModalityFusionModule()
        self.tcn = nn.ModuleList([nn.Conv1d(384, 384, 3, padding=d, dilation=d) for d in [1, 2, 4, 8]])
        self.bn = nn.ModuleList([nn.BatchNorm1d(384) for _ in range(4)])
        self.droppath = nn.ModuleList([DropPath(0.1) for _ in range(4)])
        self.dropout_tcn = nn.Dropout(0.4)
        self.attn_pool = nn.Linear(384, 1)
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.hand_classifier = nn.Linear(128, num_classes)
        self.numeric_classifier = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, is_numeric=None):
        fused, hands = self.fusion(x, is_numeric)
        for conv, bn, dp in zip(self.tcn, self.bn, self.droppath):
            res = fused
            fused = F.relu(bn(conv(fused)))
            fused = self.dropout_tcn(fused)
            if fused.size(2) == res.size(2): fused = res + dp(fused)
        weights = F.softmax(self.attn_pool(fused.transpose(1, 2)), dim=1)
        x_p = (fused * weights.transpose(1, 2)).sum(dim=2)
        num_logits = self.numeric_classifier(x_p).squeeze(-1)
        m_feat = F.relu(self.bn_fc1(self.fc1(x_p)))
        m_logits = self.fc2(F.dropout(m_feat, 0.5, self.training))
        h_logits = self.hand_classifier(hands.mean(dim=2))
        return m_logits + 0.3 * h_logits, num_logits

# --- 학습 로직 ---

def train_epoch(model, loader, criterion, optimizer, device, scaler, scheduler=None):
    model.train(); tl, cor, tot = 0, 0, 0
    bin_crit = nn.BCEWithLogitsLoss()
    for s, l, n in tqdm(loader, desc='Training'):
        s, l, n = s.to(device), l.to(device), n.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out, num_logits = model(s, is_numeric=n)
            num_prob = torch.sigmoid(num_logits).detach()
            sample_weight = 1.0 + 1.5 * num_prob
            losses = criterion(out, l, per_sample=True)
            loss = (losses * sample_weight).mean()
            num_loss = bin_crit(num_logits, n)
            total_loss = loss + 0.3 * num_loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
        tl += total_loss.item(); _, p = out.max(1); tot += l.size(0); cor += p.eq(l).sum().item()
    return tl/len(loader), 100.*cor/tot

def evaluate(model, loader, criterion, device):
    model.eval(); tl, c1, c5, tot = 0, 0, 0, 0
    with torch.no_grad():
        for s, l, n in loader:
            s, l, n = s.to(device), l.to(device), n.to(device)
            out, _ = model(s, is_numeric=n)
            tl += criterion(out, l).item(); tot += l.size(0)
            c1 += out.max(1)[1].eq(l).sum().item()
            c5 += (out.topk(5, 1)[1] == l.unsqueeze(-1)).any(1).sum().item()
    return tl/len(loader), 100.*c1/tot, 100.*c5/tot

def collate_fn(batch):
    p = pad_sequence([it[0] for it in batch], batch_first=True)
    l = torch.tensor([it[1] for it in batch], dtype=torch.long)
    n = torch.tensor([it[2] for it in batch], dtype=torch.float)
    return p, l, n

if __name__ == "__main__":
    def is_colab():
        try: return 'google.colab' in str(get_ipython())
        except: return False
    BASE_DIR = '/content/drive/MyDrive/processed' if is_colab() else 'processed'
    LR = 0.001 

    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts), label_map = load_datasets(processed_dir=BASE_DIR)
    
    tr_ds_raw = SignDataset(X_tr, y_tr, label_map, face_weight=2.0)
    tr_ds = PreprocessedSignDataset(tr_ds_raw, split='train', augment=True)
    
    s_weights = [1.5 if is_num else 1.0 for is_num in tr_ds_raw.is_numeric for _ in range(6)]
    sampler = WeightedRandomSampler(s_weights, num_samples=len(s_weights), replacement=True)
    
    tr_loader = DataLoader(tr_ds, batch_size=64, sampler=sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    vl_loader = DataLoader(PreprocessedSignDataset(SignDataset(X_vl, y_vl, label_map), augment=False), batch_size=128, collate_fn=collate_fn)
    ts_loader = DataLoader(PreprocessedSignDataset(SignDataset(X_ts, y_ts, label_map), augment=False), batch_size=128, collate_fn=collate_fn)

    model = TCNClassifier(len(label_map)).to('cuda')
    counts = torch.bincount(y_tr, minlength=len(label_map)).float()
    cw = 1.0 / (torch.sqrt(counts) + 1e-6); cw /= cw.mean()
    criterion = ImprovedFocalLoss(alpha=cw.to('cuda'), gamma=1.5, smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=epochs)
    early_stop = EarlyStopping(patience=14, verbose=True, path='best_ver2_3.pt')
    scaler = torch.amp.GradScaler('cuda')

    print(f"Train Dataset Size: {len(tr_ds)}, Loader Steps: {len(tr_loader)}")
    for epoch in range(epochs):
        t_l, t_a = train_epoch(model, tr_loader, criterion, optimizer, 'cuda', scaler, scheduler)
        v_l, v_a, v_5 = evaluate(model, vl_loader, criterion, 'cuda')
        print(f"Epoch {epoch+1}/{epochs}\nTrain Loss: {t_l:.4f}, Train Acc: {t_a:.2f}%\nVal Loss: {v_l:.4f}, Val Acc: {v_a:.2f}%, Val Top-5: {v_5:.2f}%")
        early_stop(v_l, model)
        if early_stop.early_stop: break
    
    if os.path.exists('best_ver2_3.pt'):
        model.load_state_dict(torch.load('best_ver2_3.pt'))
    _, ts_a, ts_5 = evaluate(model, ts_loader, criterion, 'cuda')
    print(f"\nFinal Test Acc: {ts_a:.2f}%, Top-5: {ts_5:.2f}%")
