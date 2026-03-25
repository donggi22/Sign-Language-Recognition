import os
import json
import torch
from collections import Counter

def clean_labels(y_list):
    """라벨에서 \n을 제거하고 공백을 정리합니다."""
    return [l.split('\n')[0].strip() for l in y_list]

def load_shards(processed_dir, split):
    """지정된 분할(train, val, test)의 모든 .pt 파일에서 라벨을 로드합니다."""
    shard_dir = os.path.join(processed_dir, split)
    all_labels = []
    if not os.path.exists(shard_dir):
        print(f"경고: {shard_dir} 디렉토리가 존재하지 않습니다.")
        return []
    
    for fname in sorted(os.listdir(shard_dir)):
        if fname.endswith('.pt'):
            try:
                shard = torch.load(os.path.join(shard_dir, fname), map_location='cpu')
                all_labels.extend(shard['labels'])
            except Exception as e:
                print(f"파일 로드 오류 ({fname}): {e}")
    return all_labels

def generate():
    processed_dir = 'processed'
    top_n_classes = 994
    
    print("모든 분할에서 라벨을 로드하여 공통 클래스(994개)를 식별하는 중...")
    y_train_str = load_shards(processed_dir, 'train')
    y_val_str   = load_shards(processed_dir, 'val')
    y_test_str  = load_shards(processed_dir, 'test')
    
    # 라벨 정리
    y_train_clean = clean_labels(y_train_str)
    y_val_clean   = clean_labels(y_val_str)
    y_test_clean  = clean_labels(y_test_str)
    
    # 각 분할별 고유 클래스 집합
    s_train = set(y_train_clean)
    s_val   = set(y_val_clean)
    s_test  = set(y_test_clean)
    
    # 세 분할 모두에 존재하는 공통 클래스 추출
    common_classes = s_train & s_val & s_test
    
    # Train 빈도수 기준으로 정렬
    counts = Counter(y_train_clean)
    top_common = [label for label, _ in counts.most_common() if label in common_classes]
    
    # 최종 994개 클래스 확정 (model_train.py와 동일한 로직)
    final_labels = top_common[:top_n_classes]
    
    # 매핑 데이터 생성
    id_to_label = {i: label for i, label in enumerate(final_labels)}
    label_to_id = {label: i for i, label in enumerate(final_labels)}
    
    output_data = {
        "count": len(final_labels),                    # 전체 클래스 수
        "labels": final_labels,                        # 인덱스 순서대로 정렬된 리스트
        "id_to_label": id_to_label,                    # 인덱스 -> 단어 매핑
        "label_to_id": label_to_id                     # 단어 -> 인덱스 매핑
    }
    
    output_path = 'final_label_map.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"성공: {len(final_labels)}개의 라벨을 '{output_path}'에 저장했습니다.")
    print("이 매핑은 현재 model_train.py의 학습 설정과 정확히 일치합니다.")

if __name__ == "__main__":
    generate()
