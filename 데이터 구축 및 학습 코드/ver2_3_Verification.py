import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# ver2_3에서 클래스 및 함수 임포트
from ver2_3 import TCNClassifier, SignDataset, PreprocessedSignDataset, load_datasets, collate_fn, ImprovedFocalLoss

def analyze_ver2_3(model_path, base_dir='processed'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 데이터 로드 (테스트 셋)
    print("Loading test dataset...")
    _, _, (X_ts, y_ts), label_map = load_datasets(processed_dir=base_dir, top_n_classes=994)
    ts_ds = PreprocessedSignDataset(SignDataset(X_ts, y_ts, label_map), augment=False)
    ts_loader = DataLoader(ts_ds, batch_size=128, collate_fn=collate_fn, num_workers=2)
    
    # 2. 모델 로드
    print(f"Loading model from {model_path}...")
    # hand_dim=136 for ver2_3 (68*2)
    model = TCNClassifier(len(label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    is_num_list = []
    
    print(f"\n=== ver2_3 성능 검증 시작 (강건한 지숫자 인식 모델) ===")
    
    with torch.no_grad():
        for s, l, n in tqdm(ts_loader, desc="Evaluating"):
            s, l, n = s.to(device), l.to(device), n.to(device)
            outputs, _ = model(s, is_numeric=n)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(l.cpu().numpy())
            is_num_list.extend(n.cpu().numpy().astype(bool))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    is_num_list = np.array(is_num_list)
    
    # 3. 전체 통계
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    print(f"\n[전체 결과]")
    print(f"Accuracy: {correct/total:.2%} ({correct}/{total})")
    
    # 4. 유형별 분석 (지숫자 vs 일반)
    num_mask = is_num_list
    char_mask = ~is_num_list
    
    if num_mask.any():
        num_acc = (all_preds[num_mask] == all_labels[num_mask]).sum() / num_mask.sum()
        print(f"지숫자(Numeric) Accuracy: {num_acc:.2%} ({all_preds[num_mask].sum()}/{num_mask.sum()})")
    
    if char_mask.any():
        char_acc = (all_preds[char_mask] == all_labels[char_mask]).sum() / char_mask.sum()
        print(f"일반(Character) Accuracy: {char_acc:.2%} ({all_preds[char_mask].sum()}/{char_mask.sum()})")

    # 5. 지숫자 전용 Confusion Matrix 시각화
    idx_to_label = {v: k for k, v in label_map.items()}
    numeric_indices = [i for i, lbl in idx_to_label.items() if any(c.isdigit() for c in lbl)]
    
    if numeric_indices:
        # 지숫자 샘플만 추출
        num_labels_subset = all_labels[num_mask]
        num_preds_subset = all_preds[num_mask]
        
        # 지숫자 클래스들만 대상으로 필터링
        # (실제 라벨이 지숫자인 것 중 모델이 예측한 값도 지숫자 범위 내에 있는 것 위주로 분석)
        unique_num_labels = np.unique(num_labels_subset)
        cm = confusion_matrix(num_labels_subset, num_preds_subset, labels=unique_num_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', 
                    xticklabels=[idx_to_label[i] for i in unique_num_labels],
                    yticklabels=[idx_to_label[i] for i in unique_num_labels])
        plt.title('Numeric Signs Confusion Matrix (V2.3)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_v2_3.png')
        print(f"\nConfusion Matrix 저장 완료: confusion_matrix_v2_3.png")

        # Top 10 Errors (지숫자 내에서)
        errors = num_labels_subset != num_preds_subset
        if errors.any():
            err_pairs = list(zip(num_labels_subset[errors], num_preds_subset[errors]))
            most_common_errs = Counter(err_pairs).most_common(10)
            print("\n[상위 10개 지숫자 오답 조합]")
            for (act, pre), count in most_common_errs:
                print(f"  {idx_to_label[act]} -> {idx_to_label[pre]} : {count}회")

if __name__ == "__main__":
    def is_colab():
        try: return 'google.colab' in str(get_ipython())
        except: return False
    
    BASE_DIR = '/content/drive/MyDrive/processed' if is_colab() else 'processed'
    MODEL_FILE = 'best_ver2_3.pt'
    
    if os.path.exists(MODEL_FILE):
        analyze_ver2_3(MODEL_FILE, BASE_DIR)
    else:
        print(f"Info: {MODEL_FILE} 파일을 찾을 수 없습니다.")
