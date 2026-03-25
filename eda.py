import os, json
import torch
from collections import Counter

# 데이터셋 로드 및 클래스 필터링 함수
def load_datasets(processed_dir='processed', top_n_classes=1024):

    def load_shards(split):
        shard_dir = os.path.join(processed_dir, split)
        all_pose, all_left, all_right, all_face = [], [], [], []
        all_labels = []

        for fname in sorted(os.listdir(shard_dir)):
            if not fname.endswith('.pt'):
                continue

            shard = torch.load(os.path.join(shard_dir, fname))

            all_pose.append(shard['pose'])
            all_left.append(shard['left_hand'])
            all_right.append(shard['right_hand'])
            all_face.append(shard['face'])

            # strip 유지 (이건 필요)
            all_labels.extend([l.strip() for l in shard['labels']])

        X = {
            'pose':       torch.cat(all_pose),
            'left_hand':  torch.cat(all_left),
            'right_hand': torch.cat(all_right),
            'face':       torch.cat(all_face),
        }

        return X, all_labels


    # label_map
    with open(os.path.join(processed_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    X_train, y_train_str = load_shards('train')
    X_val,   y_val_str   = load_shards('val')
    X_test,  y_test_str  = load_shards('test')


    # 🔥 공통 클래스 추출 및 라벨 정리 (\n 제거)
    def clean_labels(y_list):
        return [l.split('\n')[0].strip() for l in y_list]

    y_train_clean = clean_labels(y_train_str)
    y_val_clean   = clean_labels(y_val_str)
    y_test_clean  = clean_labels(y_test_str)

    s_train = set(y_train_clean)
    s_val   = set(y_val_clean)
    s_test  = set(y_test_clean)

    common_classes = s_train & s_val & s_test

    # 🔥 top-N 클래스 (Train 빈도순, 공통 클래스만)
    counts = Counter(y_train_clean)
    top_common = [label for label, _ in counts.most_common() if label in common_classes]
    top_labels = {label: i for i, label in enumerate(top_common[:top_n_classes])}


    # 🔥 빠른 필터링 (torch index)
    def filter_data(X, y_str_orig):
        # 원본 라벨을 정리해서 매핑
        mapped = []
        for y in y_str_orig:
            cleaned = y.split('\n')[0].strip()
            mapped.append(top_labels.get(cleaned, -1))
        
        mapped_tensor = torch.tensor(mapped, dtype=torch.long)
        mask = mapped_tensor != -1
        
        indices = mask.nonzero(as_tuple=True)[0]

        filtered_X = {k: v[indices] for k, v in X.items()}
        filtered_y = mapped_tensor[indices]

        return filtered_X, filtered_y


    X_train, y_train = filter_data(X_train, y_train_str)
    X_val,   y_val   = filter_data(X_val, y_val_str)
    X_test,  y_test  = filter_data(X_test, y_test_str)


    # 🔥 디버깅 출력
    print(f"Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    print(f"Classes (train 기준): {len(top_labels)}")

    print("Train 클래스:", len(torch.unique(y_train)))
    print("Val 클래스:", len(torch.unique(y_val)))
    print("Test 클래스:", len(torch.unique(y_test)))


    return (X_train, y_train), (X_val, y_val), (X_test, y_test), top_labels

processed_dir = 'processed'

with open(os.path.join(processed_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
    label_map = json.load(f)

print("전체 정의된 클래스 수:", len(label_map))

# ===== 실제 실행 =====
(train_X, y_train), (val_X, y_val), (test_X, y_test), top_labels = load_datasets(processed_dir)

# ===== 클래스 개수 확인 =====
print("Train 클래스 수:", len(set(y_train)))
print("Val 클래스 수:", len(set(y_val)))
print("Test 클래스 수:", len(set(y_test)))
print("전체 (합집합):", len(set(y_train) | set(y_val) | set(y_test)))