import streamlit as st
from PIL import Image
import base64
from utils.func import gif_paths, get_base64_of_file, go_to_step

def app():
    # col1, col2, col3 = st.columns([0.9, 1, 1])
    # with col2:
    #     if st.button("🔙뒤로"):
    #         go_to_step(4)

    try:
        gif_path = gif_paths[0]
        data_url = get_base64_of_file(gif_path)
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" style="display: block; margin: 0 auto;" alt="Welcome GIF">',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"GIF 파일을 찾을 수 없습니다: {gif_paths[0]}")
    st.markdown("<h2 style='text-align: center; color: #333;'>길찾기 서비스입니다.<br> 아래의 버튼을 눌러주세요.</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📖 이용 안내", key="guide_button", use_container_width=True):
            go_to_step(1)

    with col2:
        if st.button("▶ 바로 이용", key="start_button", use_container_width=True):
            go_to_step(2)