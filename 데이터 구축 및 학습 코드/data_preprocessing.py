import os
import json
import numpy as np
import torch
from tqdm import tqdm

# 기존 데이터 처리 함수
def convert_keypoints_to_numpy(keypoint_data):
    pose = np.array(keypoint_data['pose'], dtype=np.float32)
    left_hand = np.array(keypoint_data['left_hand'], dtype=np.float32)
    right_hand = np.array(keypoint_data['right_hand'], dtype=np.float32)
    face = np.array(keypoint_data['face'], dtype=np.float32)
    return {
        'pose': pose,
        'left_hand': left_hand,
        'right_hand': right_hand,
        'face': face
    }

def create_sequence_array(segment_frames):
    num_frames = len(segment_frames)
    pose_size = 25 * 3
    hand_size = 21 * 3
    face_size = 70 * 3

    pose_seq = np.zeros((num_frames, pose_size), dtype=np.float32)
    left_hand_seq = np.zeros((num_frames, hand_size), dtype=np.float32)
    right_hand_seq = np.zeros((num_frames, hand_size), dtype=np.float32)
    face_seq = np.zeros((num_frames, face_size), dtype=np.float32)

    for i, frame in enumerate(segment_frames):
        pose_seq[i] = frame['pose']
        left_hand_seq[i] = frame['left_hand']
        right_hand_seq[i] = frame['right_hand']
        face_seq[i] = frame['face']

    return {
        'pose': pose_seq,
        'left_hand': left_hand_seq,
        'right_hand': right_hand_seq,
        'face': face_seq
    }

def standardize_sequence(sequence_data, target_length=150):
    """시퀀스 길이를 표준화하는 함수"""
    def resample_array(arr, target_len):
        current_len = arr.shape[0]
        if current_len == target_len:
            return arr
        indices = np.linspace(0, current_len - 1, target_len, dtype=int)
        return arr[indices]

    standardized = {}
    for key in ['pose', 'left_hand', 'right_hand', 'face']:
        if sequence_data[key].shape[0] > 0:
            standardized[key] = resample_array(sequence_data[key], target_length)
        else:
            feature_dim = sequence_data[key].shape[1] if len(sequence_data[key].shape) > 1 else 0
            standardized[key] = np.zeros((target_length, feature_dim))
    return standardized

def get_matched_files_generator(base_path='1.Training/'):
    keypoint_base = os.path.join(base_path, 'keypoint')
    if not os.path.exists(keypoint_base):
        print(f"Base keypoint path not found: {keypoint_base}")
        return

    # 실제 존재하는 사람 폴더 목록 가져오기 (숫자 형태인 것만)
    person_folders = sorted([f for f in os.listdir(keypoint_base) if os.path.isdir(os.path.join(keypoint_base, f))])
    
    for person_folder in person_folders:
        keypoint_path = os.path.join(keypoint_base, person_folder)
        morpheme_path = os.path.join(base_path, 'morpheme', person_folder)

        if not os.path.exists(keypoint_path) or not os.path.exists(morpheme_path):
            print(f"Skipping folder {person_folder} - path does not exist")
            continue

        for sign_folder in os.listdir(keypoint_path):
            try:
                morpheme_file = os.path.join(morpheme_path, f"{sign_folder}_morpheme.json")
                if not os.path.exists(morpheme_file):
                    continue

                with open(morpheme_file, 'r', encoding='utf-8') as f:
                    morpheme_data = json.load(f)
                    duration = float(morpheme_data['metaData']['duration'])

                sign_path = os.path.join(keypoint_path, sign_folder)
                if not os.path.exists(sign_path):
                    continue

                keypoint_files = sorted(
                    [f for f in os.listdir(sign_path) if f.endswith('_keypoints.json')],
                    key=lambda x: int(x.split('_')[-2])
                )

                if not keypoint_files:
                    continue

                total_frames = len(keypoint_files)
                fps = total_frames / duration

                for segment in morpheme_data['data']:
                    try:
                        start_time = float(segment['start'])
                        end_time = float(segment['end'])
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)

                        if start_frame >= total_frames or end_frame >= total_frames:
                            continue

                        segment_frames = []
                        for frame_idx in range(start_frame, end_frame + 1):
                            if frame_idx >= len(keypoint_files):
                                break

                            kp_file = keypoint_files[frame_idx]
                            try:
                                with open(os.path.join(sign_path, kp_file), 'r', encoding='utf-8') as f:
                                    keypoint_data = json.load(f)
                                    frame_data = {
                                        'pose': keypoint_data['people'][0]['pose_keypoints_2d'] if isinstance(keypoint_data['people'], list) and len(keypoint_data['people']) > 0 else keypoint_data['people']['pose_keypoints_2d'],
                                        'left_hand': keypoint_data['people'][0]['hand_left_keypoints_2d'] if isinstance(keypoint_data['people'], list) and len(keypoint_data['people']) > 0 else keypoint_data['people']['hand_left_keypoints_2d'],
                                        'right_hand': keypoint_data['people'][0]['hand_right_keypoints_2d'] if isinstance(keypoint_data['people'], list) and len(keypoint_data['people']) > 0 else keypoint_data['people']['hand_right_keypoints_2d'],
                                        'face': keypoint_data['people'][0]['face_keypoints_2d'] if isinstance(keypoint_data['people'], list) and len(keypoint_data['people']) > 0 else keypoint_data['people']['face_keypoints_2d']
                                    }
                                    frame_data = convert_keypoints_to_numpy(frame_data)
                                    segment_frames.append(frame_data)
                            except (json.JSONDecodeError, KeyError, IndexError) as e:
                                # JSON 구조가 다를 수 있으므로 예외 처리 강화
                                continue

                        if segment_frames:
                            sequence_data = create_sequence_array(segment_frames)
                            standardized = standardize_sequence(sequence_data)
                            
                            yield {
                                'file_id': sign_folder,
                                'person_id': person_folder,
                                'sequence': standardized,
                                'label': segment['attributes'][0]['name'],
                                'start_time': start_time,
                                'end_time': end_time
                            }

                    except Exception as e:
                        continue
            except Exception as e:
                continue

def save_shard(shard_data, base_dir, split_name, shard_idx):
    """여러 샘플을 하나의 샤드(.pt) 파일로 묶어서 저장"""
    os.makedirs(base_dir, exist_ok=True)
    
    # 텐서 스택 생성
    # 각 샘플: {'pose': (150, 75), 'left_hand': (150, 63), ...}
    batch_pose = torch.stack([torch.from_numpy(d['sequence']['pose']).float() for d in shard_data])
    batch_left = torch.stack([torch.from_numpy(d['sequence']['left_hand']).float() for d in shard_data])
    batch_right = torch.stack([torch.from_numpy(d['sequence']['right_hand']).float() for d in shard_data])
    batch_face = torch.stack([torch.from_numpy(d['sequence']['face']).float() for d in shard_data])
    
    # 라벨 리스트
    labels = [d['label'] for d in shard_data]
    
    # 메타데이터 (필요한 경우)
    metadata = [{
        'file_id': d['file_id'],
        'person_id': d['person_id'],
        'start_time': d['start_time'],
        'end_time': d['end_time']
    } for d in shard_data]

    shard = {
        'pose': batch_pose,
        'left_hand': batch_left,
        'right_hand': batch_right,
        'face': batch_face,
        'labels': labels,
        'metadata': metadata
    }
    
    save_path = os.path.join(base_dir, f"{split_name}_shard_{shard_idx:03d}.pt")
    torch.save(shard, save_path)
    return len(shard_data)

def process_all_data(shard_size=200):
    """모든 데이터를 처리하고 Train/Val/Test 샤드 단위로 저장"""
    output_dir = 'processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # 출력 디렉토리 설정
    dirs = {
        'train': os.path.join(output_dir, 'train'),
        'val': os.path.join(output_dir, 'val'),
        'test': os.path.join(output_dir, 'test')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 각 세트별 상태 관리
    shards = {
        'train': {'data': [], 'idx': 0, 'total': 0},
        'val': {'data': [], 'idx': 0, 'total': 0},
        'test': {'data': [], 'idx': 0, 'total': 0}
    }
    
    labels_set = set()
    
    print(f"Starting optimized 3-way split (shard_size={shard_size})...")
    
    # 처리할 소스 폴더 목록
    source_paths = ['1.Training/', '2.Validation/']
    
    for base_path in source_paths:
        if not os.path.exists(base_path):
            continue
            
        desc = f"Processing {base_path}"
        for data in tqdm(get_matched_files_generator(base_path=base_path), desc=desc):
            person_id = int(data['person_id'])
            labels_set.add(data['label'])
            
            # --- 분할 전략 (Person ID 기준) ---
            # 1~15: Train (약 15,000개)
            # 16~17: Val   (약 2,000개)
            # 18~19: Test  (약 2,000개)
            if person_id <= 15:
                split = 'train'
            elif person_id <= 17:
                split = 'val'
            else:
                split = 'test'
            
            # 해당 split의 샤드 버퍼에 추가
            shards[split]['data'].append(data)
            
            # 샤드 사이즈 도달 시 저장
            if len(shards[split]['data']) >= shard_size:
                count = save_shard(shards[split]['data'], dirs[split], split, shards[split]['idx'])
                shards[split]['total'] += count
                shards[split]['idx'] += 1
                shards[split]['data'] = []
            
    # 남은 데이터들 저장
    for split in shards:
        if shards[split]['data']:
            count = save_shard(shards[split]['data'], dirs[split], split, shards[split]['idx'])
            shards[split]['total'] += count
            shards[split]['idx'] += 1

    # 라벨 맵 저장
    label_list = sorted(list(labels_set))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=4)
            
    print(f"\n최종 요약:")
    for split in shards:
        print(f"{split.upper()} 세트: {shards[split]['total']} 샘플 ({shards[split]['idx']} 샤드)")
    print(f"총 클래스 수: {len(label_to_id)}")
    print(f"데이터가 '{output_dir}/' 폴더 내 train/val/test로 구성되었습니다.")

if __name__ == "__main__":
    process_all_data()
