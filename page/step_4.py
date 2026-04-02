import streamlit as st
from PIL import Image
import base64
from utils.func import gif_paths, get_base64_of_file, go_to_step, image_paths

def app():
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔙뒤로"):
            go_to_step(3)

    destination = st.session_state.get("destination", "경복궁")
    try:
        gif_path = gif_paths[3]
        data_url = get_base64_of_file(gif_path)
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" style="display: block; margin: 0 auto;" alt="Guide GIF">',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"GIF 파일을 찾을 수 없습니다: {gif_paths[3]}")
    
    st.markdown(f"<h2 style='text-align: center; color: #333;'>5번 출구로 나오면 계단을 따라 올라가세요.</h2>", unsafe_allow_html=True)
    
    # 이미지 경로
    image_path = image_paths[1]

    # CSS를 사용해 가운데 정렬
    st.markdown(
        """
        <style>
        .centered-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;  # 필요에 따라 너비 조정
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 로컬 이미지를 base64로 인코딩
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # HTML로 base64 이미지를 가운데 정렬
    st.markdown(
        f'<img src="data:image/jpeg;base64,{encoded_string}" class="centered-image"><br>',
        unsafe_allow_html=True
    )
    
    st.write()

    col1, col2, col3 = st.columns([0.75, 1, 1])
    with col2:
        if st.button("🔜다음"):
            go_to_step(5)