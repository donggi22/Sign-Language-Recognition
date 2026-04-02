import streamlit as st
from PIL import Image
import base64
from utils.func import gif_paths, get_base64_of_file, go_to_step, image_paths

def app():
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔙뒤로"):
            go_to_step(2)

    destination = st.session_state.get("destination", "청와대")
    try:
        gif_path = gif_paths[5]
        data_url = get_base64_of_file(gif_path)
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" style="display: block; margin: 0 auto;" alt="Guide GIF">',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"GIF 파일을 찾을 수 없습니다: {gif_paths[5]}")
    
    st.markdown(f"<h2 style='text-align: center; color: #333;'>{destination}(으)로 가시려면 <br>경복궁역 4번 출구로 나와 청와대까지 도보 15분 이동<b2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.75, 1, 1])
    with col2:
        if st.button("🔚종료"):
            go_to_step(0)