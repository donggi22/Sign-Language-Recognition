import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
# ver2_2에서 클래스 및 함수 임포트
from ver2_2 import TCNClassifier, SignDataset, PreprocessedSignDataset, load_datasets, collate_fn, ImprovedFocalLoss

def analyze_ver2_2(model_path, base_dir='processed'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 데이터 로드 (테스트 셋)
    print("Loading test dataset...")
    _, _, (X_ts, y_ts), label_map = load_datasets(processed_dir=base_dir, top_n_classes=994)
    ts_ds = PreprocessedSignDataset(SignDataset(X_ts, y_ts, label_map), augment=False)
    ts_loader = DataLoader(ts_ds, batch_size=128, collate_fn=collate_fn, num_workers=2)
    
    # 2. 모델 로드
    print(f"Loading model from {model_path}...")
    model = TCNClassifier(len(label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total = 0
    correct = 0
    top5_correct = 0
    
    # 유형별 통계
    num_total = 0
    num_correct = 0
    char_total = 0
    char_correct = 0
    
    # 이진 분류(지숫자 여부) 통계
    binary_correct = 0
    
    # 오답 기록용
    total_errors = 0
    num_errors = 0 
    char_errors = 0 

    print(f"\n=== ver2_2 성능 검증 시작 (과적합 억제 모델) ===")
    
    with torch.no_grad():
        for s, l, n in tqdm(ts_loader, desc="Evaluating"):
            s, l, n = s.to(device), l.to(device), n.to(device)
            
            # 인퍼런스
            outputs, numeric_logits = model(s, is_numeric=n)
            
            _, predicted = outputs.max(1)
            _, top5 = outputs.topk(5, 1)
            
            num_prob = torch.sigmoid(numeric_logits)
            is_num_pred = num_prob > 0.5
            is_num_actual = n > 0.5
            
            # 기본 통계
            batch_size = l.size(0)
            total += batch_size
            correct += predicted.eq(l).sum().item()
            top5_correct += (top5 == l.unsqueeze(-1)).any(1).sum().item()
            binary_correct += (is_num_pred == is_num_actual).sum().item()
            
            # 유형별 분석
            for i in range(batch_size):
                actual_is_num = is_num_actual[i].item()
                is_correct = (predicted[i] == l[i]).item()
                
                if actual_is_num: # 실제 지숫자
                    num_total += 1
                    if is_correct:
                        num_correct += 1
                    else:
                        num_errors += 1
                        total_errors += 1
                else: # 실제 일반수어
                    char_total += 1
                    if is_correct:
                        char_correct += 1
                    else:
                        char_errors += 1
                        total_errors += 1

    # 리포트 출력
    print("\n" + "="*60)
    print(f" [ver2_2 최종 분석 리포트]")
    print(f" 1. 전체 Accuracy (Top-1): {correct/total:.2%} ({correct}/{total})")
    print(f" 2. 전체 Accuracy (Top-5): {top5_correct/total:.2%} ({top5_correct}/{total})")
    print(f" 3. 지숫자 판별 정답률 (Binary Type Acc): {binary_correct/total:.2%} ({binary_correct}/{total})")
    print("-" * 60)
    
    num_acc = num_correct / num_total if num_total > 0 else 0
    char_acc = char_correct / char_total if char_total > 0 else 0
    
    print(f" [지숫자 (Numeric)] 정확도: {num_acc:.2%} ({num_correct}/{num_total})")
    print(f" [일반 (Character)] 정확도: {char_acc:.2%} ({char_correct}/{char_total})")
    print("-" * 60)
    
    if total_errors > 0:
        print(f" * 전체 오답 수: {total_errors}")
        print(f" * 오답 중 지숫자 비중: {num_errors/total_errors:.1%} ({num_errors})")
        print(f" * 오답 중 일반수어 비중: {char_errors/total_errors:.1%} ({char_errors})")
    else:
        print(" * 오답이 없습니다!")
    print("="*60)

if __name__ == "__main__":
    def is_colab():
        try: return 'google.colab' in str(get_ipython())
        except: return False
    
    BASE_DIR = '/content/drive/MyDrive/processed' if is_colab() else 'processed'
    MODEL_FILE = 'best_ver2_2.pt'
    
    if os.path.exists(MODEL_FILE):
        try:
            analyze_ver2_2(MODEL_FILE, BASE_DIR)
        except Exception as e:
            print(f"\n[Error] 실행 중 오류가 발생했습니다: {e}")
            print(f"현재 BASE_DIR 설정: {BASE_DIR}")
    else:
        print(f"Info: {MODEL_FILE} 파일을 찾을 수 없습니다. 학습 완료 후 이 스크립트를 실행하여 성능을 검증하세요.")