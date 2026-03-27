import torch
if __name__ == "__main__":
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

import os
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
from itertools import cycle

torch.backends.cudnn.benchmark = True

# --- 유틸리티 및 전처리 ---

def is_numeric_label(label):
    if re.search(r'\d', label): return True
    ordinals = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덞째", "하옵째", "열째"]
    if any(label.startswith(o) for o in ordinals): return True
    numeric_suffixes = ["회", "시간", "분", "시", "일", "월", "달", "년", "km", "명", "살", "호선"]
    if any(label.endswith(s) for s in numeric_suffixes): return True
    return False

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='best_ver2_1.pt'):
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
        if not os.path.exists(shard_dir): return None, None
        all_p, all_l, all_r, all_f, all_lbl = [], [], [], [], []
        for f in sorted(os.listdir(shard_dir)):
            if not f.endswith('.pt'): continue
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
    def __init__(self, X, y, label_map, augment=False):
        self.X, self.y, self.label_map, self.augment = X, y, label_map, augment
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.is_numeric = [is_numeric_label(self.idx_to_label[int(ly)]) for ly in y]
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        p, lh, rh, f = self.X['pose'][idx], self.X['left_hand'][idx], self.X['right_hand'][idx], self.X['face'][idx]
        is_num = self.is_numeric[idx]
        feat = torch.cat([p, lh, rh, f * (0.5 if is_num else 3.0)], dim=1)
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        if self.augment:
            if is_num: feat = self._apply_digit_augmentation(feat)
            else: feat = self._apply_standard_augmentation(feat)
        return feat.float(), int(self.y[idx]), int(is_num)

    def _apply_digit_augmentation(self, f):
        strategy = random.randint(0, 4)
        if strategy == 1: # Speed Perturbation
            scale = random.uniform(0.7, 1.3)
            l = max(1, int(len(f) * scale))
            res = F.interpolate(f.transpose(0,1).unsqueeze(0), size=l, mode='linear', align_corners=True).squeeze(0).transpose(0,1)
            return F.pad(res, (0,0,0,max(0,150-l)))[:150]
        if strategy == 2: # Hand Jitter
            jitter = torch.zeros_like(f); jitter[:, 75:201] = torch.randn(f.size(0), 126)*0.05
            return f + jitter
        if strategy == 3: # Frame Dropout
            mask = torch.rand(f.size(0)) > 0.15
            if mask.any(): return f[mask].repeat(2, 1)[:150]
        return f

    def _apply_standard_augmentation(self, f):
        strategy = random.randint(0, 3)
        if strategy == 1: return f + torch.randn_like(f)*0.02
        if strategy == 2: m = f.clone(); m[:, ::2] = -m[:, ::2]; return m
        return f

# --- 모델 구조 (Optimized Multi-Head) ---

class TemporalExpert(nn.Module):
    def __init__(self, in_dim, out_dim, dilation_max=8):
        super().__init__()
        layers = []
        d = 1
        while d <= dilation_max:
            layers.extend([nn.Conv1d(in_dim if d==1 else out_dim, out_dim, 3, padding=d, dilation=d), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout1d(0.2)])
            d *= 2
        self.tcn = nn.Sequential(*layers)
    def forward(self, x): return self.tcn(x)

class TaskSeparatedTCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.p_enc = nn.Sequential(nn.Conv1d(75, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.h_enc = nn.Sequential(nn.Conv1d(126, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.f_enc = nn.Sequential(nn.Conv1d(210, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        
        self.shared_tcn = TemporalExpert(384, 256, dilation_max=4)
        self.type_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        
        # [REINFORCED] Char Head with Expert
        self.char_expert = TemporalExpert(256, 256, dilation_max=4)
        self.char_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes))
        
        # Digit Head (Heavy Duty)
        self.digit_expert = TemporalExpert(256, 256, dilation_max=8)
        self.digit_attn = nn.MultiheadAttention(256, 4)
        self.digit_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes))
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        p = self.p_enc(x[:, :, :75].transpose(1,2))
        h = self.h_enc(x[:, :, 75:201].transpose(1,2))
        f = self.f_enc(x[:, :, 201:].transpose(1,2))
        
        shared = self.shared_tcn(torch.cat([p, h, f], 1))
        global_feat = self.pool(shared).squeeze(-1)
        type_logits = self.type_head(global_feat).squeeze(-1)
        
        # Char Path
        c_feat = self.char_expert(shared)
        c_final = self.pool(c_feat).squeeze(-1)
        out_char = self.char_head(c_final)
        
        # Digit Path
        d_feat = self.digit_expert(shared)
        d_seq = d_feat.permute(2, 0, 1)
        d_attn, _ = self.digit_attn(d_seq, d_seq, d_seq)
        d_final = self.pool((d_seq + d_attn).permute(1, 2, 0)).squeeze(-1)
        out_digit = self.digit_head(d_final)
        
        return out_digit, out_char, type_logits

# --- 학습 로직 (Re-Balanced Step) ---

def train_epoch(model, d_loader, c_loader, criterion, optimizer, device, scaler, num_steps):
    model.train(); tl, cor, tot = 0, 0, 0
    d_iter, c_iter = cycle(d_loader), iter(c_loader)
    bin_crit = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(range(num_steps), desc='Training')
    for _ in pbar:
        try: d_s, d_l, d_n = next(d_iter)
        except StopIteration: d_iter = cycle(d_loader); d_s, d_l, d_n = next(d_iter)
        try: c_s, c_l, c_n = next(c_iter)
        except StopIteration: break 
        
        s, l, n = torch.cat([d_s, c_s]), torch.cat([d_l, c_l]), torch.cat([d_n, c_n])
        s, l, n = s.to(device), l.to(device), n.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out_d, out_c, type_out = model(s)
            
            is_digit = (n > 0.5)
            loss = 0
            if is_digit.any(): 
                loss += 0.8 * criterion(out_d[is_digit], l[is_digit]) # Slight lower weight for digit loss
            if (~is_digit).any(): 
                loss += 1.2 * criterion(out_c[~is_digit], l[~is_digit]) # Higher weight for char loss
            loss += 0.5 * bin_crit(type_out, n)
            
        if torch.isnan(loss): continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        
        tl += loss.item(); tot += l.size(0)
        # Threshold: 0.4 (Balanced bias)
        is_num_pred = torch.sigmoid(type_out) > 0.4
        final_pred = torch.where(is_num_pred, out_d.max(1)[1], out_c.max(1)[1])
        cor += final_pred.eq(l).sum().item()
        pbar.set_postfix({'loss': f'{tl/(pbar.n+1):.4f}', 'acc': f'{100.*cor/tot:.2f}%'})
        
    return tl/num_steps, 100.*cor/tot

def evaluate(model, loader, criterion, device):
    model.eval(); tl, c1, c5, tot = 0, 0, 0, 0
    bin_crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for s, l, n in loader:
            s, l, n = s.to(device), l.to(device), n.to(device)
            out_d, out_c, type_out = model(s)
            is_digit = (n > 0.5)
            loss = 0
            if is_digit.any(): loss += criterion(out_d[is_digit], l[is_digit])
            if (~is_digit).any(): loss += criterion(out_c[~is_digit], l[~is_digit])
            loss += 0.5 * bin_crit(type_out, n)
            tl += loss.item(); tot += l.size(0)
            is_num_pred = torch.sigmoid(type_out) > 0.4
            final_out = torch.where(is_num_pred.unsqueeze(-1), out_d, out_c)
            c1 += final_out.max(1)[1].eq(l).sum().item()
            c5 += (final_out.topk(5, 1)[1] == l.unsqueeze(-1)).any(1).sum().item()
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
    NUM_WORKERS = 2 if is_colab() else 4
    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts), label_map = load_datasets(processed_dir=BASE_DIR)
    
    tr_ds = SignDataset(X_tr, y_tr, label_map, augment=True)
    d_idx = [i for i, n in enumerate(tr_ds.is_numeric) if n]
    c_idx = [i for i, n in enumerate(tr_ds.is_numeric) if not n]
    
    # [RE-BALANCED] 25% Digits (32) : 75% Chars (96)
    d_loader = DataLoader(Subset(tr_ds, d_idx), batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    c_loader = DataLoader(Subset(tr_ds, c_idx), batch_size=96, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    
    vl_loader = DataLoader(SignDataset(X_vl, y_vl, label_map), batch_size=128, collate_fn=collate_fn)
    ts_loader = DataLoader(SignDataset(X_ts, y_ts, label_map), batch_size=128, collate_fn=collate_fn)

    model = TaskSeparatedTCN(len(label_map)).to('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    early_stop = EarlyStopping(patience=12, verbose=True, path='best_ver2_1.pt')
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(50):
        t_l, t_a = train_epoch(model, d_loader, c_loader, criterion, optimizer, 'cuda', scaler, len(c_loader))
        v_l, v_a, v_5 = evaluate(model, vl_loader, criterion, 'cuda')
        print(f"Epoch {epoch+1}/50\nTrain Loss: {t_l:.4f}, Train Acc: {t_a:.2f}%\nVal Loss: {v_l:.4f}, Val Acc: {v_a:.2f}%, Val Top-5: {v_5:.2f}%")
        scheduler.step(v_l); early_stop(v_l, model)
        if early_stop.early_stop: break
    
    model.load_state_dict(torch.load('best_ver2_1.pt'))
    _, ts_a, ts_5 = evaluate(model, ts_loader, criterion, 'cuda')
    print(f"\nFinal Test Acc: {ts_a:.2f}%, Top-5: {ts_5:.2f}%")
