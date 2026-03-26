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
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# 지숫자(숫자 관련 수어) 여부 판별 함수
def is_numeric_label(label):
    if re.search(r'\d', label):
        return True
    ordinals = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덞째", "하옵째", "열째"]
    if any(label.startswith(o) for o in ordinals):
        return True
    numeric_suffixes = ["회", "시간", "분", "시", "일", "월", "달", "년", "km", "명", "살", "호선"]
    if any(label.endswith(s) for s in numeric_suffixes):
        return True
    return False

# Focal Loss (어려운 샘플에 가중치를 더 두는 손실 함수)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 데이터셋 로드 및 클래스 필터링 함수
def load_datasets(processed_dir='processed/', top_n_classes=994):
    def load_shards(split):
        shard_dir = os.path.join(processed_dir, split)
        all_pose, all_left, all_right, all_face, all_labels = [], [], [], [], []

        for fname in sorted(os.listdir(shard_dir)):
            if not fname.endswith('.pt'):
                continue
            shard = torch.load(os.path.join(shard_dir, fname))
            all_pose.append(shard['pose'])
            all_left.append(shard['left_hand'])
            all_right.append(shard['right_hand'])
            all_face.append(shard['face'])
            all_labels.extend([l.strip() for l in shard['labels']])

        X = {
            'pose':       torch.cat(all_pose),
            'left_hand':  torch.cat(all_left),
            'right_hand': torch.cat(all_right),
            'face':       torch.cat(all_face),
        }
        return X, all_labels

    with open(os.path.join(processed_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    X_train, y_train_str = load_shards('train')
    X_val,   y_val_str   = load_shards('val')
    X_test,  y_test_str  = load_shards('test')

    def clean_labels(y_list):
        return [l.split('\n')[0].strip() for l in y_list]

    y_train_clean = clean_labels(y_train_str)
    y_val_clean   = clean_labels(y_val_str)
    y_test_clean  = clean_labels(y_test_str)

    s_train = set(y_train_clean)
    s_val   = set(y_val_clean)
    s_test  = set(y_test_clean)
    common_classes = s_train & s_val & s_test

    counts = Counter(y_train_clean)
    top_common = [label for label, _ in counts.most_common() if label in common_classes]
    top_labels = {label: new_id for new_id, label in enumerate(top_common[:top_n_classes])}

    def filter_data(X, y_str_orig):
        mapped = []
        for y in y_str_orig:
            cleaned = y.split('\n')[0].strip()
            mapped.append(top_labels.get(cleaned, -1))
        
        mapped_tensor = torch.tensor(mapped, dtype=torch.long)
        mask = mapped_tensor != -1
        filtered_X = {k: v[mask] for k, v in X.items()}
        filtered_y = mapped_tensor[mask]

        return filtered_X, filtered_y

    X_train, y_train = filter_data(X_train, y_train_str)
    X_val,   y_val   = filter_data(X_val,   y_val_str)
    X_test,  y_test  = filter_data(X_test,  y_test_str)

    print(f"Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    print(f"Classes: {len(top_labels)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), top_labels

# Dataset 클래스 (지숫자 강화 및 형태소 정보 보존)
class SignDataset(Dataset):
    def __init__(self, X, y, label_map, face_weight=3.0):
        self.X = X
        self.y = y
        self.label_to_idx = label_map
        self.idx_to_label = {int(idx): label for label, idx in label_map.items()}
        self.face_weight = face_weight
        self.is_numeric = [is_numeric_label(self.idx_to_label[int(label_idx)]) for label_idx in self.y]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 지숫자일 경우 얼굴 보다는 손모양에 집중하도록 가중치 조절
        current_face_weight = self.face_weight * 0.5 if self.is_numeric[idx] else self.face_weight
        
        face_features = self.X['face'][idx] * current_face_weight
        other_features = torch.cat([
            self.X['pose'][idx],
            self.X['left_hand'][idx],
            self.X['right_hand'][idx]
        ], dim=1)

        features = torch.cat([other_features, face_features], dim=1)
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        features = (features - features.mean()) / (features.std() + 1e-8)

        label = int(self.y[idx])
        morpheme = {'data': [{'attributes': [{'name': self.idx_to_label[label]}]}]}
        
        return features.float(), label, morpheme

class PreprocessedSignDataset(Dataset):
    def __init__(self, dataset, split='train', augment=True):
        self.dataset = dataset
        self.split = split
        self.augment = augment and split == 'train'
        self.label_to_idx = dataset.label_to_idx
        self.idx_to_label = dataset.idx_to_label
        self.aug_factor = 6 if self.augment else 1

    def __len__(self):
        return len(self.dataset) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % len(self.dataset)
        aug_idx  = idx // len(self.dataset)
        features, label, morpheme = self.dataset[orig_idx]
        if self.augment:
            features = self._apply_augmentation(features, aug_idx)
        return features, label, morpheme

    def _apply_augmentation(self, features, aug_idx):
        if aug_idx == 0:
            return features
        elif aug_idx == 1:
            return self.time_scale_augment(features)
        elif aug_idx == 2:
            return self.noise_augment(features)
        elif aug_idx == 3:
            return self.spatial_transform(features)
        elif aug_idx == 4:
            return self.mirror_augment(features)
        elif aug_idx == 5:
            return self.rotation_augment(features)
        return features

    def time_scale_augment(self, sequence, scale_range=(0.85, 1.15)):
        scale = random.uniform(*scale_range)
        length = max(1, int(len(sequence) * scale))
        sequence = sequence.transpose(0, 1).unsqueeze(0)
        resampled = F.interpolate(sequence, size=length, mode='linear', align_corners=True)
        if length < 150:
            resampled = F.pad(resampled, (0, 150 - length), mode='constant', value=0)
        elif length > 150:
            resampled = resampled[:, :, :150]
        return resampled.squeeze(0).transpose(0, 1)

    def noise_augment(self, sequence, noise_level=0.03):
        noise = torch.randn_like(sequence) * noise_level
        return sequence + noise

    def spatial_transform(self, sequence, max_shift=0.05):
        shift = torch.randn_like(sequence) * max_shift
        return sequence + shift

    def mirror_augment(self, sequence):
        mirrored = sequence.clone()
        mirrored[:, ::2] = -mirrored[:, ::2]
        return mirrored

    def rotation_augment(self, sequence, max_angle=10):
        device = sequence.device
        angle = (torch.rand(1, device=device) * 2 - 1) * max_angle
        rad = torch.deg2rad(angle)
        cos = torch.cos(rad)
        sin = torch.sin(rad)
        theta = torch.stack([
            torch.stack([cos, -sin]),
            torch.stack([sin,  cos])
        ]).squeeze()
        coords = sequence.view(sequence.shape[0], -1, 3)
        xy = coords[:, :, :2]
        coords[:, :, :2] = torch.matmul(xy, theta)
        return coords.view_as(sequence)

# 크로스 어텐션 모듈 (개선됨: 양손 특징 보존)
class CrossModalAttention(nn.Module):
    def __init__(self, pose_dim, hand_dim, face_dim, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.pose_proj = nn.Linear(pose_dim, embed_dim)
        self.hand_proj = nn.Linear(hand_dim, embed_dim)
        self.face_proj = nn.Linear(face_dim, embed_dim)

        self.attn_pose = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm_pose = nn.LayerNorm(embed_dim)

        self.attn_hand = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm_hand = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(pose_dim + hand_dim + face_dim + embed_dim * 2, 384)

    def forward(self, pose_features, left_hand_features, right_hand_features, face_features):
        pose = pose_features.permute(2, 0, 1)
        hands = torch.cat([left_hand_features, right_hand_features], dim=1).permute(2, 0, 1)
        face = face_features.permute(2, 0, 1)

        p_emb = self.pose_proj(pose)
        h_emb = self.hand_proj(hands)
        f_emb = self.face_proj(face)

        context = torch.cat([h_emb, f_emb], dim=0)
        attn_out_pose, _ = self.attn_pose(p_emb, context, context)
        p_res = self.norm_pose(p_emb + self.dropout(attn_out_pose))

        attn_out_hand, _ = self.attn_hand(h_emb, h_emb, h_emb)
        h_res = self.norm_hand(h_emb + self.dropout(attn_out_hand))

        p_orig = pose.permute(1, 0, 2)
        h_orig = hands.permute(1, 0, 2)
        f_orig = face.permute(1, 0, 2)
        p_res = p_res.permute(1, 0, 2)
        h_res = h_res.permute(1, 0, 2)

        fused = torch.cat([p_orig, h_orig, f_orig, p_res, h_res], dim=2)
        fused = self.out_proj(fused)
        return fused.permute(0, 2, 1)

# 멀티모달 특징 융합 모듈
class ModalityFusionModule(nn.Module):
    def __init__(self, pose_dim=75, left_hand_dim=63, right_hand_dim=63, face_dim=210):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(pose_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.left_hand_encoder = nn.Sequential(
            nn.Conv1d(left_hand_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.right_hand_encoder = nn.Sequential(
            nn.Conv1d(right_hand_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.face_encoder = nn.Sequential(
            nn.Conv1d(face_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # hand_dim = 128(left) + 128(right) = 256
        self.cross_attention = CrossModalAttention(128, 256, 256)

    def forward(self, x):
        pose = x[:, :, :75]
        left_hand = x[:, :, 75:138]
        right_hand = x[:, :, 138:201]
        face = x[:, :, 201:]
        
        pose_features = self.pose_encoder(pose.transpose(1, 2))
        left_hand_features = self.left_hand_encoder(left_hand.transpose(1, 2))
        right_hand_features = self.right_hand_encoder(right_hand.transpose(1, 2))
        face_features = self.face_encoder(face.transpose(1, 2))
        
        return self.cross_attention(pose_features, left_hand_features, right_hand_features, face_features)

# TCNClassifier
class TCNClassifier(nn.Module):
    def __init__(self, input_size=411, num_classes=150):
        super().__init__()
        self.fusion_module = ModalityFusionModule()
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(384, 384, kernel_size=3, padding=dilation, dilation=dilation)
            for dilation in [1, 2, 4, 8]
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(384) for _ in range(4)])
        self.dropout_tcn = nn.Dropout(0.3)
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'fc' in name and 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.fusion_module(x)
        for conv, bn in zip(self.tcn_layers, self.bn_layers):
            residual = x
            x = F.relu(bn(conv(x)))
            x = self.dropout_tcn(x)
            if x.size(2) == residual.size(2):
                x = x + residual
        x = x.mean(dim=2)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    morphemes = [item[2]['data'][0]['attributes'][0]['name'] for item in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels, morphemes

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for sequences, labels, _ in tqdm(dataloader, desc='Training'):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    return avg_loss, 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
    with torch.no_grad():
        for sequences, labels, _ in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()
            _, top5_pred = outputs.topk(5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    correct_top5 += 1
    return total_loss/len(dataloader), 100.*correct_top1/total, 100.*correct_top5/total

def is_colab():
    try:
        return 'google.colab' in str(get_ipython())
    except NameError:
        return False

if __name__ == "__main__":
    if is_colab(): # colab 환경
        BASE_DIR = '/content/drive/MyDrive/processed'
        NUM_WORKERS = 2
    else: # local GPU 환경
        BASE_DIR = 'processed'
        NUM_WORKERS = 4
    
    top_n_classes = 994
    BATCH_SIZE = 128
    LR = 0.001

    print(f"Loading datasets with {top_n_classes} classes...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map = load_datasets(processed_dir=BASE_DIR, top_n_classes=top_n_classes)

    train_dataset = SignDataset(X_train, y_train, label_map, face_weight=3.0)
    val_dataset = SignDataset(X_val, y_val, label_map, face_weight=3.0)
    test_dataset = SignDataset(X_test, y_test, label_map, face_weight=3.0)

    # 지숫자 샘플 가중치 계산 (샘플러용)
    sample_weights = [3.0 if is_num else 1.0 for is_num in train_dataset.is_numeric]
    expanded_weights = []
    for w in sample_weights: 
        expanded_weights.extend([w] * 6)
    train_sampler = WeightedRandomSampler(expanded_weights, num_samples=len(expanded_weights), replacement=True)

    processed_train_dataset = PreprocessedSignDataset(train_dataset, split='train', augment=True)
    processed_val_dataset = PreprocessedSignDataset(val_dataset, split='validation', augment=False)
    processed_test_dataset = PreprocessedSignDataset(test_dataset, split='test', augment=False)

    print("\n전처리 결과:")
    print(f"원본 훈련 데이터 크기: {len(train_dataset)}")
    print(f"전처리된 훈련 데이터 크기: {len(processed_train_dataset)}")
    print(f"원본 검증 데이터 크기: {len(val_dataset)}")
    print(f"전처리된 검증 데이터 크기: {len(processed_val_dataset)}")
    print(f"원본 테스트 데이터 크기: {len(test_dataset)}")
    print(f"전처리된 테스트 데이터 크기: {len(processed_test_dataset)}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(processed_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(processed_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(processed_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    model = TCNClassifier(num_classes=len(label_map)).to(device)

    # Focal Loss 설정 (지숫자 성능 향상을 위해 어려운 샘플에 집중)
    y_train_long = y_train.long().to(device)
    class_counts = torch.bincount(y_train_long, minlength=len(label_map)).float()
    class_weights = 1.0 / (torch.sqrt(class_counts) + 1e-6)
    class_weights = class_weights / class_weights.mean()
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='best_tcn_improved_model.pt')

    # History 딕셔너리 초기화
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5_acc': []
    }

    scaler = torch.amp.GradScaler('cuda')
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_acc, val_top5_acc = evaluate(model, val_loader, criterion, device)

        # History에 값 추가
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5 Acc: {val_top5_acc:.2f}%")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load('best_tcn_improved_model.pt'))
    test_loss, test_acc, test_top5_acc = evaluate(model, test_loader, criterion, device)
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Top-5 Acc: {test_top5_acc:.2f}%")
    print("Training and evaluation completed!")