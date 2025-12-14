import streamlit as st
from PIL import Image
import base64
from utils.func import gif_paths, get_base64_of_file, go_to_step

def app():
    col1, col2, = st.columns([1, 1])
    with col1:
        if st.button("🔙뒤로"):
            go_to_step(0)

    st.markdown("<div class='main-title'>📖 이용 안내</div>", unsafe_allow_html=True)
    st.write("""
    - ### 본 서비스는 수어를 인식하여 번역하는 키오스크입니다.
    - ### 웹캠을 통해 수어를 입력하거나, 직접 텍스트를 입력할 수 있습니다.
    - ### 목적지를 입력하면 가장 적합한 출구를 안내해 드립니다.
    - ### '바로 이용'을 선택하면 웹캠을 활성화하거나 목적지를 입력할 수 있습니다.
    """)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🏠 홈", key="home_button", use_container_width=True):
            go_to_step(0)
    with col2:
        if st.button("🚀 바로 이용", key="start_button_guide", use_container_width=True):
            go_to_step(2)
    st.markdown("</div>", unsafe_allow_html=True)