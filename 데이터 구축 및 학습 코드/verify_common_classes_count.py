import os
import torch
from collections import Counter

def clean_label(label):
    return label.split('\n')[0].strip()

def get_labels(processed_dir, split):
    shard_dir = os.path.join(processed_dir, split)
    all_labels = []
    if not os.path.exists(shard_dir):
        print(f"Directory {shard_dir} does not exist.")
        return []
    
    for fname in sorted(os.listdir(shard_dir)):
        if fname.endswith('.pt'):
            try:
                shard = torch.load(os.path.join(shard_dir, fname), map_location='cpu')
                all_labels.extend([clean_label(l) for l in shard['labels']])
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    return all_labels

def verify():
    processed_dir = 'processed'
    print(f"Loading labels from {processed_dir}...")
    
    train_labels = get_labels(processed_dir, 'train')
    val_labels = get_labels(processed_dir, 'val')
    test_labels = get_labels(processed_dir, 'test')
    
    s_train = set(train_labels)
    s_val = set(val_labels)
    s_test = set(test_labels)
    
    intersection = s_train & s_val & s_test
    
    print("\n--- Results ---")
    print(f"Train unique classes: {len(s_train)}")
    print(f"Val unique classes:   {len(s_val)}")
    print(f"Test unique classes:  {len(s_test)}")
    print(f"Common classes (Intersection): {len(intersection)}")
    
    # if len(intersection) > 0:
    #     print("\nTop 5 frequent common classes in Train:")
    #     counts = Counter(train_labels)
    #     top_common = [label for label, _ in counts.most_common() if label in intersection]
    #     for i, label in enumerate(top_common[:5]):
    #         print(f"{i+1}. {label} ({counts[label]} samples)")

if __name__ == "__main__":
    verify()