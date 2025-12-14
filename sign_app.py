import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import cv2
import mediapipe as mp
import tempfile
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import streamlit.components.v1 as components
from collections import deque
import time
import warnings
import logging
from utils.func import morpheme_folder_path

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 모델 정의
class CrossModalAttention(nn.Module):
    def __init__(self, pose_dim, hand_dim, face_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.value_proj = nn.Linear(face_dim, 128)
        self.out_proj = nn.Linear(128 + 256 + 128, 384)

    def forward(self, pose_features, left_hand_features, right_hand_features, face_features):
        pose = pose_features.permute(2, 0, 1)
        hands = torch.cat([left_hand_features, right_hand_features], dim=1).permute(2, 0, 1)
        face = face_features.permute(2, 0, 1)
        Q = pose
        K = hands[:, :, :128]
        V = self.value_proj(face)
        attn_output, _ = self.attention(Q, K, V)
        fused = torch.cat([pose, hands, attn_output], dim=2)
        fused = self.out_proj(fused.permute(1, 0, 2))
        return fused.permute(0, 2, 1)

class ModalityFusionModule(nn.Module):
    def __init__(self, pose_dim=75, left_hand_dim=63, right_hand_dim=63, face_dim=210):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(pose_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.left_hand_encoder = nn.Sequential(
            nn.Conv1d(left_hand_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.right_hand_encoder = nn.Sequential(
            nn.Conv1d(right_hand_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.face_encoder = nn.Sequential(
            nn.Conv1d(face_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.cross_attention = CrossModalAttention(128, 128, 256)

    def forward(self, x):
        pose = x[:, :, :75]
        left_hand = x[:, :, 75:138]
        right_hand = x[:, :, 138:201]
        face = x[:, :, 201:]
        pose_features = self.pose_encoder(pose.transpose(1, 2))
        left_hand_features = self.left_hand_encoder(left_hand.transpose(1, 2))
        right_hand_features = self.right_hand_encoder(right_hand.transpose(1, 2))
        face_features = self.face_encoder(face.transpose(1, 2))
        fused_features = self.cross_attention(pose_features, left_hand_features, right_hand_features, face_features)
        return fused_features

class TCNClassifier(nn.Module):
    def __init__(self, input_size=411, num_classes=995):
        super().__init__()
        self.fusion_module = ModalityFusionModule()
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(384, 384, kernel_size=3, padding=dilation*1, dilation=dilation)
            for dilation in [1, 2, 4, 8]
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(384) for _ in range(4)])
        self.dropout_tcn = nn.Dropout(0.3)
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

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

# --------------------------------------------------------------------------------
# 실시간 수어 인식 클래스
class RealTimeSignRecognizer:
    def __init__(self, model, id_to_label, buffer_size=150, movement_threshold=0.05, 
                 idle_frames=15, confidence_threshold=0.7, face_weight=3.0):
        self.model = model
        self.id_to_label = id_to_label
        self.buffer_size = buffer_size
        self.movement_threshold = movement_threshold
        self.idle_frames = idle_frames
        self.confidence_threshold = confidence_threshold
        self.face_weight = face_weight
        
        # 키포인트 버퍼 (고정 길이 큐)
        self.keypoints_buffer = deque(maxlen=buffer_size)
        
        # 동작 감지 관련 변수
        self.is_signing = False
        self.idle_counter = 0
        self.prev_keypoints = None
        self.current_sequence = []
        self.detection_start_time = None
        
        # 최근 예측 결과
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_prediction_time = None
        
        # 예측 이력
        self.prediction_history = []
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_keypoints(self, frame, holistic_results):
        # MediaPipe 결과를 OpenPose 형식으로 변환
        keypoints_data = convert_mediapipe_to_openpose(holistic_results, frame.shape)
        
        # 키포인트 추출 및 형식 변환
        frame_data = {
            'pose': keypoints_data['people']['pose_keypoints_2d'],
            'left_hand': keypoints_data['people']['hand_left_keypoints_2d'],
            'right_hand': keypoints_data['people']['hand_right_keypoints_2d'],
            'face': keypoints_data['people']['face_keypoints_2d']
        }
        
        return frame_data
    
    def calculate_movement(self, current, previous):
        try:
            # 손 키포인트의 움직임 계산
            left_hand_current = np.array(current['left_hand']).reshape(-1, 3)
            right_hand_current = np.array(current['right_hand']).reshape(-1, 3)
            
            left_hand_prev = np.array(previous['left_hand']).reshape(-1, 3)
            right_hand_prev = np.array(previous['right_hand']).reshape(-1, 3)
            
            # 신뢰도가 높은 키포인트만 사용 (임계값 상향 조정)
            left_conf_mask = left_hand_current[:, 2] > 0.6  # 신뢰도 임계값 상향
            right_conf_mask = right_hand_current[:, 2] > 0.6  # 신뢰도 임계값 상향
            
            # 신뢰도 높은 키포인트가 최소 5개 이상 있는지 확인
            left_valid = np.sum(left_conf_mask) >= 5
            right_valid = np.sum(right_conf_mask) >= 5
            
            left_diff = 0
            right_diff = 0
            
            # 신뢰도 높은 키포인트가 충분할 때만 움직임 계산
            if left_valid:
                left_diff = np.mean(np.sqrt(np.sum(
                    (left_hand_current[left_conf_mask, :2] - left_hand_prev[left_conf_mask, :2])**2, axis=1
                )))
            
            if right_valid:
                right_diff = np.mean(np.sqrt(np.sum(
                    (right_hand_current[right_conf_mask, :2] - right_hand_prev[right_conf_mask, :2])**2, axis=1
                )))
            
            # 움직임 값이 너무 작으면 0으로 설정 (노이즈 제거)
            movement = max(left_diff, right_diff)
            if movement < 2.0:  # 픽셀 단위의 최소 움직임 임계값
                movement = 0
                
            return movement
        except Exception as e:
            print(f"Movement calculation error: {e}")
            return 0
    
    def convert_keypoints_to_numpy(self, keypoint_data):
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
    
    def create_sequence_array(self, segment_frames):
        num_frames = len(segment_frames)
        pose_size = 25 * 3
        hand_size = 21 * 3
        face_size = 70 * 3

        pose_seq = np.zeros((num_frames, pose_size), dtype=np.float32)
        left_hand_seq = np.zeros((num_frames, hand_size), dtype=np.float32)
        right_hand_seq = np.zeros((num_frames, hand_size), dtype=np.float32)
        face_seq = np.zeros((num_frames, face_size), dtype=np.float32)

        for i, frame in enumerate(segment_frames):
            np_frame = self.convert_keypoints_to_numpy(frame)
            pose_seq[i] = np_frame['pose']
            left_hand_seq[i] = np_frame['left_hand']
            right_hand_seq[i] = np_frame['right_hand']
            face_seq[i] = np_frame['face']

        return {
            'pose': pose_seq,
            'left_hand': left_hand_seq,
            'right_hand': right_hand_seq,
            'face': face_seq
        }
    
    def standardize_sequence(self, sequence_data, target_length=150):
        def resample_array(arr, target_len):
            current_len = arr.shape[0]
            if current_len == 0:
                # 빈 배열 처리
                feature_dim = arr.shape[1] if len(arr.shape) > 1 else 0
                return np.zeros((target_len, feature_dim))
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
    
    def preprocess_sequence(self, sequence):
        # 시퀀스 처리 및 모델 입력 형태로 변환
        try:
            np_sequence = self.create_sequence_array(sequence)
            standardized = self.standardize_sequence(np_sequence)
            
            # 특성 결합 및 정규화
            face_features = standardized['face'] * self.face_weight
            other_features = np.concatenate([
                standardized['pose'],
                standardized['left_hand'],
                standardized['right_hand']
            ], axis=1)
            features = np.concatenate([other_features, face_features], axis=1)
            
            # 정규화
            if np.std(features) > 1e-8:
                features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, features):
        # 모델을 통한 예측
        try:
            if features is None:
                return "데이터 부족", 0.0, []
                
            input_tensor = torch.tensor(features.reshape(1, 150, 411), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred_id = torch.max(probs, dim=1)
                
                # Top-3 예측 결과 확인
                top3_prob, top3_indices = torch.topk(probs, 3, dim=1)
                top3_pred = [(self.id_to_label[str(idx.item())], prob.item()) for idx, prob in zip(top3_indices[0], top3_prob[0])]
                
                # 코드2처럼 항상 예측값 반환
                return self.id_to_label[str(pred_id.item())], confidence.item(), top3_pred
        except Exception as e:
            print(f"Prediction error: {e}")
            return "오류", 0.0, []
    
    def process_frame(self, frame, holistic_results):
        # 현재 시간 기록
        current_time = time.time()
        
        # 키포인트 추출
        keypoints = self.extract_keypoints(frame, holistic_results)
        
        # 키포인트 버퍼에 추가
        self.keypoints_buffer.append(keypoints)
        
        result = None
        confidence = 0.0
        top3_predictions = []
        
        # 이전 프레임이 있으면 움직임 계산
        if self.prev_keypoints is not None:
            movement = self.calculate_movement(keypoints, self.prev_keypoints)
            
            # 손이 감지되었는지 확인 (손의 신뢰도 값이 높은 키포인트 개수)
            left_hand_visible = np.sum(np.array(keypoints['left_hand']).reshape(-1, 3)[:, 2] > 0.6) >= 8
            right_hand_visible = np.sum(np.array(keypoints['right_hand']).reshape(-1, 3)[:, 2] > 0.6) >= 8
            hands_visible = left_hand_visible or right_hand_visible
            
            # 동작 시작 감지 - 움직임이 임계값보다 크고 손이 잘 보일 때
            if not self.is_signing and movement > self.movement_threshold and hands_visible:
                self.is_signing = True
                self.current_sequence = list(self.keypoints_buffer)[-30:]  # 최근 30프레임부터 시작
                self.idle_counter = 0
                self.detection_start_time = current_time
                print("Sign detection started")
            
            # 동작 중일 때
            elif self.is_signing:
                # 현재 키포인트 추가
                self.current_sequence.append(keypoints)
                
                # 정지 상태 감지 또는 손이 더 이상 보이지 않는 경우
                if movement < self.movement_threshold or not hands_visible:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
                
                # 일정 시간 정지하면 동작 종료로 간주
                if (self.idle_counter >= self.idle_frames and len(self.current_sequence) >= 30) or len(self.current_sequence) >= 200:
                    # 너무 짧은 시퀀스는 무시 (노이즈 억제)
                    if len(self.current_sequence) < 30:
                        self.is_signing = False
                        self.current_sequence = []
                    else:
                        # 시퀀스 처리 및 예측
                        features = self.preprocess_sequence(self.current_sequence)
                        result, confidence, top3_predictions = self.predict(features)
                        
                        # 예측 결과 저장
                        self.last_prediction = result
                        self.last_confidence = confidence
                        self.last_prediction_time = current_time
                        
                        # 예측 이력에 추가
                        self.prediction_history.append({
                            'label': result,
                            'confidence': confidence,
                            'timestamp': current_time,
                            'sequence_length': len(self.current_sequence),
                            'detection_time': current_time - self.detection_start_time
                        })
                        
                        # 최대 10개만 유지
                        if len(self.prediction_history) > 10:
                            self.prediction_history.pop(0)
                        
                        print(f"Sign detected: {result} ({confidence:.2f})")
                        
                        # 상태 초기화
                        self.is_signing = False
                        self.current_sequence = []
        
        self.prev_keypoints = keypoints
        
        # 최근 예측 결과 반환 (3초 이내의 예측 결과만)
        if self.last_prediction_time and current_time - self.last_prediction_time < 3:
            return self.last_prediction, self.last_confidence, top3_predictions, self.is_signing
        elif self.is_signing:
            return None, 0.0, [], self.is_signing
        else:
            return None, 0.0, [], False

# --------------------------------------------------------------------------------
# 키포인트 변환 및 처리 함수
def convert_mediapipe_to_openpose(results, image_shape):
    keypoints_data = {
        "version": 1.3,
        "people": {
            "person_id": -1,
            "pose_keypoints_2d": [0] * 75,
            "face_keypoints_2d": [0] * 210,
            "hand_left_keypoints_2d": [0] * 63,
            "hand_right_keypoints_2d": [0] * 63,
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        },
        "camparam": {"Intrinsics": {"data": ""}, "CameraMatrix": {"data": ""}, "Distortion": {"rows": "", "data": ""}}
    }

    height, width = image_shape[0], image_shape[1]

    pose_mapping = {
        0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 9: 24,
        15: 6, 16: 1, 17: 8, 18: 7
    }

    if results.pose_landmarks:
        for op_idx, mp_idx in pose_mapping.items():
            if mp_idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[mp_idx]
                keypoints_data["people"]["pose_keypoints_2d"][op_idx*3] = landmark.x * width
                keypoints_data["people"]["pose_keypoints_2d"][op_idx*3+1] = landmark.y * height
                keypoints_data["people"]["pose_keypoints_2d"][op_idx*3+2] = landmark.visibility

        if 11 < len(results.pose_landmarks.landmark) and 12 < len(results.pose_landmarks.landmark):
            l_shoulder = results.pose_landmarks.landmark[11]
            r_shoulder = results.pose_landmarks.landmark[12]
            neck_x = (l_shoulder.x + r_shoulder.x) / 2 * width
            neck_y = (l_shoulder.y + r_shoulder.y) / 2 * height
            neck_visibility = min(l_shoulder.visibility, r_shoulder.visibility)
            keypoints_data["people"]["pose_keypoints_2d"][1*3] = neck_x
            keypoints_data["people"]["pose_keypoints_2d"][1*3+1] = neck_y
            keypoints_data["people"]["pose_keypoints_2d"][1*3+2] = neck_visibility

        if 23 < len(results.pose_landmarks.landmark) and 24 < len(results.pose_landmarks.landmark):
            l_hip = results.pose_landmarks.landmark[23]
            r_hip = results.pose_landmarks.landmark[24]
            mid_hip_x = (l_hip.x + r_hip.x) / 2 * width
            mid_hip_y = (l_hip.y + r_hip.y) / 2 * height
            mid_hip_visibility = min(l_hip.visibility, r_hip.visibility)
            keypoints_data["people"]["pose_keypoints_2d"][8*3] = mid_hip_x
            keypoints_data["people"]["pose_keypoints_2d"][8*3+1] = mid_hip_y
            keypoints_data["people"]["pose_keypoints_2d"][8*3+2] = mid_hip_visibility

    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        face_mapping = {
            0: 127, 1: 234, 2: 93, 3: 215, 4: 58, 5: 136, 6: 149, 7: 148,
            8: 152, 9: 400, 10: 379, 11: 394, 12: 367, 13: 435, 14: 323,
            15: 454, 16: 264, 17: 70, 18: 53, 19: 52, 20: 65, 21: 221,
            22: 285, 23: 336, 24: 295, 25: 282, 26: 300, 27: 122, 29: 5,
            30: 4, 31: 98, 33: 167, 34: 164, 35: 327, 41: 110, 43: 384,
            44: 386, 46: 254, 48: 43, 54: 273, 55: 405, 56: 18, 57: 18,
            58: 182, 59: 106, 60: 43, 68: 144, 69: 374
        }
        for op_idx, mp_idx in face_mapping.items():
            if mp_idx < len(face_landmarks):
                landmark = face_landmarks[mp_idx]
                keypoints_data["people"]["face_keypoints_2d"][op_idx*3] = landmark.x * width
                keypoints_data["people"]["face_keypoints_2d"][op_idx*3+1] = landmark.y * height
                keypoints_data["people"]["face_keypoints_2d"][op_idx*3+2] = 0.85

    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            if i < 21:
                keypoints_data["people"]["hand_left_keypoints_2d"][i*3] = landmark.x * width
                keypoints_data["people"]["hand_left_keypoints_2d"][i*3+1] = landmark.y * height
                keypoints_data["people"]["hand_left_keypoints_2d"][i*3+2] = 0.8

    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            if i < 21:
                keypoints_data["people"]["hand_right_keypoints_2d"][i*3] = landmark.x * width
                keypoints_data["people"]["hand_right_keypoints_2d"][i*3+1] = landmark.y * height
                keypoints_data["people"]["hand_right_keypoints_2d"][i*3+2] = 0.8

    return keypoints_data

def draw_openpose_style_landmarks(image, keypoints_data):
    """
    OpenPose 스타일로 키포인트를 시각화하는 함수
    """
    annotated_image = image.copy()
    
    # 연결선 정의 (OpenPose 스타일)
    pose_connections = [
        (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
        (1, 8), (8, 9), (8, 12), (9, 10), (10, 11), (12, 13), (13, 14)
    ]
    
    # Pose 키포인트 (초록색)
    for i in range(0, 75, 3):
        if keypoints_data["people"]["pose_keypoints_2d"][i+2] > 0.5:
            x = int(keypoints_data["people"]["pose_keypoints_2d"][i])
            y = int(keypoints_data["people"]["pose_keypoints_2d"][i+1])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
    
    # Pose 연결선 (초록색)
    for conn in pose_connections:
        x1 = int(keypoints_data["people"]["pose_keypoints_2d"][conn[0]*3])
        y1 = int(keypoints_data["people"]["pose_keypoints_2d"][conn[0]*3+1])
        x2 = int(keypoints_data["people"]["pose_keypoints_2d"][conn[1]*3])
        y2 = int(keypoints_data["people"]["pose_keypoints_2d"][conn[1]*3+1])
        
        conf1 = keypoints_data["people"]["pose_keypoints_2d"][conn[0]*3+2]
        conf2 = keypoints_data["people"]["pose_keypoints_2d"][conn[1]*3+2]
        
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 손 키포인트 연결 정의
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8),    # 검지
        (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
        (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
        (0, 17), (17, 18), (18, 19), (19, 20)   # 소지
    ]
    
    # Left Hand 키포인트 (파란색)
    for i in range(0, 63, 3):
        if keypoints_data["people"]["hand_left_keypoints_2d"][i+2] > 0.5:
            x = int(keypoints_data["people"]["hand_left_keypoints_2d"][i])
            y = int(keypoints_data["people"]["hand_left_keypoints_2d"][i+1])
            cv2.circle(annotated_image, (x, y), 4, (0, 0, 255), -1)
    
    # Left Hand 연결선 (파란색)
    for conn in hand_connections:
        x1 = int(keypoints_data["people"]["hand_left_keypoints_2d"][conn[0]*3])
        y1 = int(keypoints_data["people"]["hand_left_keypoints_2d"][conn[0]*3+1])
        x2 = int(keypoints_data["people"]["hand_left_keypoints_2d"][conn[1]*3])
        y2 = int(keypoints_data["people"]["hand_left_keypoints_2d"][conn[1]*3+1])
        
        conf1 = keypoints_data["people"]["hand_left_keypoints_2d"][conn[0]*3+2]
        conf2 = keypoints_data["people"]["hand_left_keypoints_2d"][conn[1]*3+2]
        
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Right Hand 키포인트 (노란색)
    for i in range(0, 63, 3):
        if keypoints_data["people"]["hand_right_keypoints_2d"][i+2] > 0.5:
            x = int(keypoints_data["people"]["hand_right_keypoints_2d"][i])
            y = int(keypoints_data["people"]["hand_right_keypoints_2d"][i+1])
            cv2.circle(annotated_image, (x, y), 4, (255, 255, 0), -1)
    
    # Right Hand 연결선 (노란색)
    for conn in hand_connections:
        x1 = int(keypoints_data["people"]["hand_right_keypoints_2d"][conn[0]*3])
        y1 = int(keypoints_data["people"]["hand_right_keypoints_2d"][conn[0]*3+1])
        x2 = int(keypoints_data["people"]["hand_right_keypoints_2d"][conn[1]*3])
        y2 = int(keypoints_data["people"]["hand_right_keypoints_2d"][conn[1]*3+1])
        
        conf1 = keypoints_data["people"]["hand_right_keypoints_2d"][conn[0]*3+2]
        conf2 = keypoints_data["people"]["hand_right_keypoints_2d"][conn[1]*3+2]
        
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # Face 키포인트 (빨간색)
    for i in range(0, 210, 3):
        if keypoints_data["people"]["face_keypoints_2d"][i+2] > 0.5:
            x = int(keypoints_data["people"]["face_keypoints_2d"][i])
            y = int(keypoints_data["people"]["face_keypoints_2d"][i+1])
            cv2.circle(annotated_image, (x, y), 3, (255, 0, 0), -1)
    
    return annotated_image

# --------------------------------------------------------------------------------
# 모델 로드 및 예측 함수
@st.cache_resource
def load_model_and_mapping(model_path, mapping_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNClassifier(input_size=411, num_classes=995).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with open(mapping_path, 'r', encoding='utf-8') as f:
        id_to_label = json.load(f)
    return model, id_to_label, device

import numpy as np
import torch

def predict_sign_language(keypoints_data, model, device, id_to_label):
    try:
        # keypoints_data가 파일 경로일 경우
        if isinstance(keypoints_data, str):
            keypoints = np.loadtxt(keypoints_data, delimiter=',', skiprows=1, encoding='utf-8')
        else:
            # 이미 로드된 데이터일 경우
            keypoints = keypoints_data

        # 형상 확인 및 조정
        if keypoints.shape[0] != 150 or keypoints.shape[1] != 411:
            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(1, -1)
            if keypoints.shape[0] == 411 and keypoints.shape[1] == 150:
                keypoints = keypoints.T
            if keypoints.shape != (150, 411):
                keypoints = keypoints.reshape(150, 411)

        keypoints = keypoints.reshape(1, 150, 411)
        input_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_id = output.argmax(dim=1).item()
            pred_word = id_to_label[str(pred_id)]

        return True, pred_word
    except UnicodeDecodeError as e:
        return False, f"파일 인코딩 오류: {e}. UTF-8로 인코딩된 파일을 사용해주세요."
    except ValueError as e:
        return False, f"데이터 형상 오류: {e}. 파일은 150줄, 각 줄 411개의 값이어야 합니다."
    except Exception as e:
        return False, f"예측 중 오류: {e}"
# --------------------------------------------------------------------------------
# 실시간 웹캠 처리 함수
def process_webcam(model, id_to_label, device, 
                   movement_threshold=0.05, 
                   idle_frames=15, 
                   confidence_threshold=0.7,
                   face_weight=3.0):
    
    # 페이지 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 웹캠 화면
        st.markdown("### 웹캠 화면")
        video_frame = st.empty()
        # 로딩 메시지용 placeholder 추가
        loading_message = st.empty()
        # 초기 로딩 메시지 표시
        loading_message.markdown("<h3 style='text-align: center; color: #333;'>웹캠을 불러오는 중입니다...</h3>", unsafe_allow_html=True)
    
    with col2:
        # 인식 결과 표시 (코드2 방식으로 간소화)
        st.markdown("### 📌인식 결과")
        result_text = st.empty()
        
        # Top-3 예측 결과
        st.markdown("### 📌Top-3 예측")
        top3_texts = [st.empty() for _ in range(3)]
    
    # 종료 버튼 - 루프 외부에서 정의
    stop_button_placeholder = st.empty()
    stop_requested = stop_button_placeholder.button("웹캠 종료", key="stop_webcam_btn")
    
    # 인식기 초기화
    recognizer = RealTimeSignRecognizer(
        model=model,
        id_to_label=id_to_label,
        buffer_size=150,
        movement_threshold=movement_threshold,
        idle_frames=idle_frames,
        confidence_threshold=confidence_threshold,
        face_weight=face_weight
    )
    
    # MediaPipe Holistic 모델 초기화
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
        loading_message.empty()  # 에러 발생 시 로딩 메시지 제거
        return
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        # 이전 값 저장을 위한 변수
        prev_stop_requested = stop_requested
        
        while cap.isOpened() and not stop_requested:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                st.error("웹캠에서 프레임을 읽을 수 없습니다.")
                loading_message.empty()  # 에러 발생 시 로딩 메시지 제거
                break
            
            # 로딩 메시지 제거 (첫 프레임이 성공적으로 읽히면)
            loading_message.empty()
            
            # BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 프레임 처리
            results = holistic.process(frame_rgb)
            
            keypoints_data = convert_mediapipe_to_openpose(results, frame.shape)

            # 랜드마크 시각화
            annotated_image = draw_openpose_style_landmarks(frame_rgb, keypoints_data)
            
            # 인식기로 프레임 처리
            result, confidence, top3_pred, is_signing = recognizer.process_frame(frame, results)
            
            # 동작 감지 중이면 빨간색 테두리 추가
            if is_signing:
                annotated_image = cv2.copyMakeBorder(
                    annotated_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0]
                )
            
            # 결과 표시 (코드2 방식)
            if result:
                result_text.markdown(f"## {result}")
                
                # Top-3 예측 결과 표시
                for i, (pred, prob) in enumerate(top3_pred):
                    if i < 3:  # 최대 3개만 표시
                        top3_texts[i].markdown(f"{i+1}. **{pred}**")
            
            # 화면 업데이트
            video_frame.image(annotated_image, channels='RGB')
            
            # 종료 버튼 상태 확인 - 루프 내에서 버튼을 다시 생성하지 않음
            if st.session_state.get('stop_webcam_btn', False):
                stop_requested = True
            
            # 프레임 속도 제어
            time.sleep(0.01)
    
    finally:
        # 자원 해제
        cap.release()
        holistic.close()
        loading_message.empty()  # 종료 시 로딩 메시지 제거
        st.success("웹캠이 종료되었습니다.")
# --------------------------------------------------------------------------------
# 디버깅 이미지 렌더링 함수
def render_debug_images():
    if 'debug_images' in st.session_state and st.session_state.debug_images:
        st.write("Rendering debug images...")
        html_content = """
        <div style="overflow-x: auto; overflow-y: hidden; white-space: nowrap; padding: 10px;">
        """
        for img, caption in st.session_state.debug_images:
            pil_img = Image.fromarray(img)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f"""
            <div style="display: inline-block; margin-right: 10px; text-align: center;">
                <img src="data:image/png;base64,{img_str}" width="300">
                <p style="color: white;">{caption}</p>
            </div>
            """
        html_content += "</div>"
        sample_img = Image.fromarray(st.session_state.debug_images[0][0])
        aspect_ratio = sample_img.size[1] / sample_img.size[0]
        display_width = 300
        display_height = int(display_width * aspect_ratio)
        caption_height = 30
        total_height = display_height + caption_height + 100  # 세로 스크롤 없도록 높이 설정
        components.html(html_content, height=total_height, scrolling=True)

# --------------------------------------------------------------------------------
# 키포인트 추출 및 비디오 처리 함수
def extract_keypoints_from_video(video_file):
    temp_dir = tempfile.mkdtemp()
    file_name_base = os.path.basename(video_file.name).split('.')[0]
    output_dir = os.path.join(temp_dir, file_name_base + "_keypoints")
    os.makedirs(output_dir, exist_ok=True)

    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(temp_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.info(f"비디오 정보: {width}x{height}, {fps}fps, 총 {total_frames}프레임")

    progress_bar = st.progress(0)
    status_text = st.empty()
    keypoint_files = []
    debug_images = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            progress = int((frame_idx / total_frames) * 50)
            progress_bar.progress(progress)
            status_text.text(f"프레임 {frame_idx}/{total_frames} 키포인트 추출 중...")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            keypoints_data = convert_mediapipe_to_openpose(results, image.shape)

            output_filename = os.path.join(output_dir, f"{file_name_base}_{str(frame_idx).zfill(12)}_keypoints.json")
            with open(output_filename, 'w') as f:
                json.dump(keypoints_data, f)
            keypoint_files.append(output_filename)

            if frame_idx % 100 == 0:
                vis_img = image.copy()
                # Pose 키포인트 (초록색)
                for i in range(0, 75, 3):
                    if keypoints_data["people"]["pose_keypoints_2d"][i+2] > 0.5:
                        x = int(keypoints_data["people"]["pose_keypoints_2d"][i])
                        y = int(keypoints_data["people"]["pose_keypoints_2d"][i+1])
                        cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)
                # Face 키포인트 (빨간색)
                for i in range(0, 210, 3):
                    if keypoints_data["people"]["face_keypoints_2d"][i+2] > 0.5:
                        x = int(keypoints_data["people"]["face_keypoints_2d"][i])
                        y = int(keypoints_data["people"]["face_keypoints_2d"][i+1])
                        cv2.circle(vis_img, (x, y), 3, (255, 0, 0), -1)
                # Left Hand 키포인트 (파란색)
                for i in range(0, 63, 3):
                    if keypoints_data["people"]["hand_left_keypoints_2d"][i+2] > 0.5:
                        x = int(keypoints_data["people"]["hand_left_keypoints_2d"][i])
                        y = int(keypoints_data["people"]["hand_left_keypoints_2d"][i+1])
                        cv2.circle(vis_img, (x, y), 4, (0, 0, 255), -1)
                # Right Hand 키포인트 (노란색)
                for i in range(0, 63, 3):
                    if keypoints_data["people"]["hand_right_keypoints_2d"][i+2] > 0.5:
                        x = int(keypoints_data["people"]["hand_right_keypoints_2d"][i])
                        y = int(keypoints_data["people"]["hand_right_keypoints_2d"][i+1])
                        cv2.circle(vis_img, (x, y), 4, (255, 255, 0), -1)
                debug_images.append((cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), f"프레임 {frame_idx}"))

            frame_idx += 1

    cap.release()
    status_text.text(f"총 {frame_idx}개 프레임 처리 완료. 구간별 처리 중...")

    st.session_state.debug_images = debug_images

    segments = []
    for segment_data in get_matched_files_generator(video_file, temp_dir, file_name_base):
        face_weight = 3.0
        face_features = segment_data['sequence']['face'] * face_weight
        other_features = np.concatenate([
            segment_data['sequence']['pose'],
            segment_data['sequence']['left_hand'],
            segment_data['sequence']['right_hand']
        ], axis=1)
        features = np.concatenate([other_features, face_features], axis=1)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        txt_file_path = os.path.join(temp_dir, f"{file_name_base}_segment_{segment_data['start_frame']}_{segment_data['end_frame']}_keypoints.txt")
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write("# Normalized Keypoints for sign\n")
            for frame in features:
                frame_str = ','.join(["{:.6f}".format(x) for x in frame])
                f.write(frame_str + "\n")

        segments.append({
            'txt_file_path': txt_file_path,
            'label': segment_data['label'],
            'start_time': segment_data['start_time'],
            'end_time': segment_data['end_time']
        })

    progress_bar.progress(100)
    status_text.text("구간별 키포인트 추출 및 TXT 파일 생성 완료!")
    return segments

def get_matched_files_generator(video_file, temp_dir, file_name_base):
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    file_name = video_file.name.split('.')[0]
    person_id = file_name.split('CROWD')[1].split('_')[0]
    video_id = file_name

    # 로컬 경로로 수정
    # morpheme_folder_path = f'C:/Users/muse1/local_file/team_project/morpheme/{person_id}'
    morpheme_path = os.path.join(morpheme_folder_path, f'{person_id}', f"{video_id}_morpheme.json")

    if not os.path.exists(morpheme_path):
        st.error(f"모핌 파일을 찾을 수 없습니다: {morpheme_path}")
        return

    try:
        with open(morpheme_path, 'r', encoding='utf-8') as f:
            morpheme_data = json.load(f)
    except UnicodeDecodeError as e:
        st.error(f"파일 디코딩 오류: {e}. 파일 인코딩을 확인하세요.")
        return
    except Exception as e:
        st.error(f"모핌 파일 로드 중 오류 발생: {e}")
        return

    duration = float(morpheme_data['metaData']['duration'])

    output_dir = os.path.join(temp_dir, file_name_base + "_keypoints")
    keypoint_files = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_keypoints.json")],
        key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0])
    )

    for segment in morpheme_data['data']:
        start_time = float(segment['start'])
        end_time = float(segment['end'])
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        if start_frame >= total_frames or end_frame >= total_frames:
            continue

        segment_frames = []
        for frame_idx in range(start_frame, min(end_frame + 1, len(keypoint_files))):
            kp_file_path = keypoint_files[frame_idx]
            with open(kp_file_path, 'r') as f:
                keypoint_data = json.load(f)
                frame_data = {
                    'pose': keypoint_data['people']['pose_keypoints_2d'],
                    'left_hand': keypoint_data['people']['hand_left_keypoints_2d'],
                    'right_hand': keypoint_data['people']['hand_right_keypoints_2d'],
                    'face': keypoint_data['people']['face_keypoints_2d']
                }
                segment_frames.append(frame_data)

        if segment_frames:
            sequence_data = create_sequence_array(segment_frames)
            standardized_sequence = standardize_sequence(sequence_data)
            yield {
                'sequence': standardized_sequence,
                'label': segment['attributes'][0]['name'],
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': end_frame
            }

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
        frame_data = convert_keypoints_to_numpy(frame)
        pose_seq[i] = frame_data['pose']
        left_hand_seq[i] = frame_data['left_hand']
        right_hand_seq[i] = frame_data['right_hand']
        face_seq[i] = frame_data['face']

    return {
        'pose': pose_seq,
        'left_hand': left_hand_seq,
        'right_hand': right_hand_seq,
        'face': face_seq
    }

def standardize_sequence(sequence_data, target_length=150):
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

# --------------------------------------------------------------------------------
# 메인 애플리케이션 코드
def main():
    st.title("🌟 수어 인식 서비스 🌟")
    
    # 메뉴 설정
    menu = ["홈", "비디오 파일 분석", "키포인트 파일 분석", "실시간 웹캠 인식"]
    choice = st.sidebar.selectbox("메뉴 선택", menu)
    
    # 세션 상태 초기화
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    
    # 로컬 경로로 수정
    model_path = f'C:/Users/muse1/local_file/team_project/models/best_tcn_improved_model.pt'
    mapping_path = f'C:/Users/muse1/local_file/team_project/models/full_class_mapping.json'
    
    with st.sidebar.expander("모델 파일 경로 설정", expanded=False):
        model_path = st.text_input("모델 파일 경로", model_path)
        mapping_path = st.text_input("클래스 매핑 파일 경로", mapping_path)
    
    try:
        model, id_to_label, device = load_model_and_mapping(model_path, mapping_path)
        st.sidebar.success("모델 로드 완료!")
    except Exception as e:
        st.sidebar.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()
    
    if choice == "홈":
        st.write("비디오를 업로드하여 수어를 번역하거나 실시간 웹캠으로 수어를 인식해보세요! 🖐️")
        st.write("""
        ## 기능 소개
        
        1. **비디오 파일 분석**: 미리 녹화된 수어 영상을 업로드하여 수어를 번역합니다.
        2. **키포인트 파일 분석**: 키포인트 데이터가 있는 TXT 파일을 직접 업로드하여 분석합니다.
        3. **실시간 웹캠 인식**: 웹캠을 통해 실시간으로 수어를 인식합니다.
        
        왼쪽 메뉴에서 원하는 기능을 선택하세요.
        """)
        
        # st.image("https://via.placeholder.com/600x400.png?text=Sign+Language+Recognition", caption="수어 인식 서비스")
        
    elif choice == "비디오 파일 분석":
        st.subheader("비디오 파일 분석")
        video_file = st.file_uploader("수어 비디오 파일(.mp4)을 업로드하세요", type=["mp4"])
        
        if video_file is not None:
            st.session_state.current_file_name = video_file.name
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            video_html = f"""
            <video width="300" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
            
            if st.button("키포인트 추출 및 구간별 변환"):
                segments = extract_keypoints_from_video(video_file)
                st.session_state.segments = segments
                
                if segments:
                    for i, segment in enumerate(segments):
                        with open(segment['txt_file_path'], 'r', encoding='utf-8') as f:
                            txt_content = f.read()
                        st.download_button(
                            label=f"키포인트 TXT 다운로드",
                            data=txt_content,
                            file_name=f"{video_file.name.split('.')[0]}_segment_{segment['start_time']}_{segment['end_time']}.txt",
                            mime="text/plain",
                            key=f"download_segment_{i}"
                        )
            
            render_debug_images()
            
            if st.session_state.segments is not None and st.session_state.current_file_name == video_file.name:
                if st.button("번역하기"):
                    st.info("번역 중...")
                    for i, segment in enumerate(st.session_state.segments):
                        success, result = predict_sign_language(segment['txt_file_path'], model, device, id_to_label)
                        if success:
                            st.success(f"수어-to-텍스트 번역 완료!")
                            st.markdown(f"## 모델 예측 결과: **{result}**")
                            st.write(f"구간({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)")
                            st.write(f"모핌 레이블: **{segment['label']}**")
                            is_correct = "일치" if result == segment['label'] else "불일치"
                            st.write(f"정확 여부: **{is_correct}**")
                        else:
                            st.error(f"구간 번역 오류: {result}")
        
    elif choice == "키포인트 파일 분석":
        st.subheader("키포인트 파일 분석")
        uploaded_file = st.file_uploader("샘플 키포인트 파일(.txt)을 업로드하세요", type=["txt"])
        
        if uploaded_file is not None:
            st.write(f"업로드된 파일: {uploaded_file.name}")
            txt_content = uploaded_file.getvalue().decode('utf-8')
            temp_dir = tempfile.mkdtemp()
            txt_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            st.session_state.txt_file_path = txt_file_path
            st.session_state.txt_content = txt_content
            st.session_state.current_file_name = uploaded_file.name
            
            st.subheader("키포인트 TXT 파일 내용 (일부)")
            preview_text = txt_content[:1000] + "..." if len(txt_content) > 1000 else txt_content
            st.text_area("키포인트 데이터", preview_text, height=200)
            
            if st.button("번역하기", key="translate_txt_btn"):
                st.info("번역 중...")
                success, result = predict_sign_language(txt_file_path, model, device, id_to_label)
                
                if success:
                    st.success("번역 완료!")
                    st.markdown(f"## 모델 예측 결과: **{result}**")
                    file_name = uploaded_file.name.split('_')[0]
                    st.write(f"파일 이름에서 추정된 정답: **{file_name}**")
                    is_correct = "일치" if result == file_name else "불일치"
                    st.write(f"정확 여부: **{is_correct}**")
                else:
                    st.error(f"번역 중 오류 발생: {result}")
                    st.write("파일 형식이 올바른지 확인해주세요 (150줄, 각 줄 411개 값).")
    
    elif choice == "실시간 웹캠 인식":
        st.subheader("실시간 웹캠 수어 인식")
        
        # 웹캠 설정
        st.write("### 웹캠 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            movement_threshold = st.slider("동작 감지 임계값", 0.05, 0.5, 0.15, 0.01)
            idle_frames = st.slider("정지 프레임 수", 5, 30, 10, 1)
            
        with col2:
            confidence_threshold = st.slider("예측 신뢰도 임계값", 0.5, 0.95, 0.7, 0.05)
            face_weight = st.slider("얼굴 특징 가중치", 1.0, 5.0, 3.0, 0.1)
        
        # 초기화 및 시작 버튼
        if st.button("웹캠 시작"):
            process_webcam(
                model=model,
                id_to_label=id_to_label,
                device=device,
                movement_threshold=movement_threshold,
                idle_frames=idle_frames,
                confidence_threshold=confidence_threshold,
                face_weight=face_weight
            )

if __name__ == "__main__":
    main()