import streamlit as st
from PIL import Image
import base64
import os
from pathlib import Path

# 현재 스크립트 파일의 디렉토리 (kiosk 디렉토리)
script_dir = Path(__file__).parent

parent_dir = script_dir.parent


gif_paths = {
        0: fr"{parent_dir}\gif_images\1. '길 찾기 서비스' 입니다. 아래의 버튼을 눌러주세요..gif",
        1: fr"{parent_dir}\gif_images\2. 다음 버튼을 누루고 궁금하신 '장소'를 알려주세요..gif",
        2: fr"{parent_dir}\gif_images\3. 경복궁역 5번 출구로 나가세요..gif",
        3: fr"{parent_dir}\gif_images\4. 5번 출구로 나오면 계단을 따라 올라가세요..gif",
        4: fr"{parent_dir}\gif_images\5. 100m 앞 정면에 경복궁 매표소가 있습니다..gif",
        5: fr"{parent_dir}\gif_images\6. 경복궁역 4번 출구로 나와 청와대까지 도보 15분 이동.gif"
        }

image_paths = {
        0: fr"{parent_dir}\images\gyeongbokgung_way02.jpg",
        1: fr"{parent_dir}\images\gyeongbokgung_way05.jpg",
        2: fr"{parent_dir}\images\gyeongbokgung_way07.jpg",
        }

model_path = parent_dir / "models" / "best_tcn_improved_model.pt"
mapping_path = parent_dir / "models" / "full_class_mapping.json"
morpheme_folder_path = parent_dir / "morpheme"

# GIF 파일을 Base64로 변환
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        contents = f.read()
    return base64.b64encode(contents).decode("utf-8")

# 버튼 클릭 이벤트
def go_to_step(step):
    st.session_state.step = step
    st.session_state.gif_key += 1
    if step == 0:  # step 0으로 돌아가면 인식 결과 초기화
        st.session_state.recognized_results = []
    st.rerun()