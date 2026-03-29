import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from collections import Counter

# ver2_4에서 클래스 및 함수 임포트
from ver2_4 import TCNClassifier, SignDataset, PreprocessedSignDataset, load_datasets, collate_fn

def analyze_ver2_4(model_path, base_dir='processed'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 데이터 로드
    print("Loading test dataset...")
    _, _, (X_ts, y_ts), label_map = load_datasets(processed_dir=base_dir, top_n_classes=994)
    ts_ds_full = SignDataset(X_ts, y_ts, label_map)
    ts_loader_full = DataLoader(PreprocessedSignDataset(ts_ds_full, augment=False), 
                                batch_size=128, collate_fn=collate_fn)
    
    # 2. 모델 로드
    print(f"Loading model from {model_path}...")
    model = TCNClassifier(len(label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 평가 함수 ---
    def run_eval(loader, desc):
        all_preds, all_labels, is_num_list = [], [], []
        with torch.no_grad():
            for s, l, n in tqdm(loader, desc=desc):
                s, l, n = s.to(device), l.to(device), n.to(device)
                outputs, _ = model(s, is_numeric=n)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(l.cpu().numpy())
                is_num_list.extend(n.cpu().numpy().astype(bool))
        return np.array(all_preds), np.array(all_labels), np.array(is_num_list)

    # 3. 전체 샘플 평가 (1,995개)
    print(f"\n=== [1] 전체 샘플 평가 (Total Samples) ===")
    preds_f, labels_f, nums_f = run_eval(ts_loader_full, "Full Evaluation")
    
    acc_f = (preds_f == labels_f).mean()
    print(f"전체 Accuracy: {acc_f:.2%} ({ (preds_f == labels_f).sum() }/{len(labels_f)})")
    
    if nums_f.any():
        num_acc = (preds_f[nums_f] == labels_f[nums_f]).mean()
        print(f"지숫자(Numeric) Accuracy: {num_acc:.2%} ({ (preds_f[nums_f] == labels_f[nums_f]).sum() }/{nums_f.sum()})")
    
    # 4. 클래스당 1개 샘플 평가 (994개) - ver1.10 방식
    print(f"\n=== [2] 클래스별 대표 샘플 평가 (Per-Class, Match ver1.10) ===")
    
    # 클래스별로 인덱스 하나씩만 추출
    unique_classes = np.unique(y_ts)
    selected_indices = []
    for cls in unique_classes:
        idx = np.where(y_ts == cls)[0][0] # 첫 번째 샘플 선택
        selected_indices.append(idx)
    
    X_ts_sub = {k: v[selected_indices] for k, v in X_ts.items()}
    y_ts_sub = y_ts[selected_indices]
    
    ts_ds_sub = SignDataset(X_ts_sub, y_ts_sub, label_map)
    ts_loader_sub = DataLoader(PreprocessedSignDataset(ts_ds_sub, augment=False), 
                                batch_size=128, collate_fn=collate_fn)
    
    preds_s, labels_s, nums_s = run_eval(ts_loader_sub, "Class-wise Evaluation")
    
    acc_s = (preds_s == labels_s).mean()
    print(f"클래스 기준 Accuracy: {acc_s:.2%} ({ (preds_s == labels_s).sum() }/{len(labels_s)})")
    
    if nums_s.any():
        num_acc_s = (preds_s[nums_s] == labels_s[nums_s]).mean()
        print(f"지숫자 클래스 Accuracy: {num_acc_s:.2%} ({ (preds_s[nums_s] == labels_s[nums_s]).sum() }/{nums_s.sum()})")

    # 5. 오답 분석 및 Heatmap
    idx_to_label = {v: k for k, v in label_map.items()}
    if nums_f.any():
        num_labels_subset = labels_f[nums_f]
        num_preds_subset = preds_f[nums_f]
        unique_num_labels = np.unique(num_labels_subset)
        
        cm = confusion_matrix(num_labels_subset, num_preds_subset, labels=unique_num_labels)
        plt.figure(figsize=(12, 10))
        # 한글 깨짐 방지를 위해 인덱스로 표시하거나 폰트 설정 무시
        sns.heatmap(cm, annot=False, cmap='YlGnBu')
        plt.title('Numeric Signs Confusion Matrix (V2.4)')
        plt.savefig('confusion_matrix_v2_4.png')
        print(f"\nConfusion Matrix 저장됨: confusion_matrix_v2_4.png")

        # Top Errors
        errors = num_labels_subset != num_preds_subset
        if errors.any():
            err_pairs = list(zip(num_labels_subset[errors], num_preds_subset[errors]))
            most_common = Counter(err_pairs).most_common(10)
            print("\n[지숫자 주요 오답 조합]")
            for (act, pre), count in most_common:
                print(f"  {idx_to_label[act]} -> {idx_to_label[pre]} : {count}회")

if __name__ == "__main__":
    def is_colab():
        try: return 'google.colab' in str(get_ipython())
        except: return False
    BASE_DIR = '/content/drive/MyDrive/processed' if is_colab() else 'processed'
    MODEL_FILE = 'best_ver2_4.pt'
    if os.path.exists(MODEL_FILE):
        analyze_ver2_4(MODEL_FILE, BASE_DIR)
    else:
        print(f"Info: {MODEL_FILE} 파일을 찾을 수 없습니다.")