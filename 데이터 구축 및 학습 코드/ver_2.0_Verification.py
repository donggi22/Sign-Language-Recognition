import torch
from tqdm import tqdm
from ver_2_0 import TaskSeparatedTCN, SignDataset, load_datasets, collate_fn # ver_2.0 파일에서 불러오기

def analyze_ver2_0(model_path, base_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 데이터 로드
    _, _, (X_ts, y_ts), label_map = load_datasets(processed_dir=base_dir, top_n_classes=994)
    ts_loader = torch.utils.data.DataLoader(SignDataset(X_ts, y_ts, label_map), batch_size=128, collate_fn=collate_fn)
    
    # 2. 모델 로드
    model = TaskSeparatedTCN(len(label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total, errors = 0, 0
    num_total, num_errors = 0, 0 # 지숫자 통계
    char_total, char_errors = 0, 0 # 일반수어 통계
    type_correct = 0 # 이진 분류(지숫자 여부) 자체를 맞춘 정도

    print(f"\n=== ver_2.0 전수 오답 분석 시작 ===")
    with torch.no_grad():
        for s, l, n in tqdm(ts_loader):
            s, l, n = s.to(device), l.to(device), n.to(device)
            out_d, out_c, type_out = model(s)
            
            # 인퍼런스 결정 (학습 때와 동일하게 0.2 threshold)
            is_num_pred = torch.sigmoid(type_out) > 0.2
            final_out = torch.where(is_num_pred.unsqueeze(-1), out_d, out_c)
            _, pr = final_out.max(1)
            
            # 실제 지숫자 여부
            is_num_actual = n > 0.5
            
            total += l.size(0)
            errors += (pr != l).sum().item()
            type_correct += (is_num_pred == is_num_actual).sum().item()
            
            # 분석용 필터
            num_mask = is_num_actual
            char_mask = ~is_num_actual
            
            num_total += num_mask.sum().item()
            num_errors += ((pr != l) & num_mask).sum().item()
            
            char_total += char_mask.sum().item()
            char_errors += ((pr != l) & char_mask).sum().item()

    # 리포트 출력
    print("\n" + "="*50)
    print(f" [최종 분석 리포트]")
    print(f" 1. 전체 정확도: {(total-errors)/total:.2%}")
    print(f" 2. 지숫자 판별 정답률(Binary): {type_correct/total:.2%}")
    print("-" * 50)
    print(f" [지숫자(Numeric)] 정확도: {(num_total-num_errors)/num_total:.2%} ({num_total-num_errors}/{num_total})")
    print(f" [일반(Char)] 정확도: {(char_total-char_errors)/char_total:.2%} ({char_total-char_errors}/{char_total})")
    print("-" * 50)
    print(f" * 오답 중 지숫자 비중: {num_errors/errors:.1%}")
    print("="*50)

if __name__ == "__main__":
    # 코랩이면 경로 수정 필요
    BASE_DIR = 'processed' 
    analyze_ver2_0('best_ver2.0.pt', BASE_DIR)