import streamlit as st
from pages import home, video, keypoint, webcam, kiosk
from streamlit_option_menu import option_menu

# 통합된 CSS 스타일 적용
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #000000, #000000);
            background-attachment: fixed;
            height: 100vh;
        }
        body, h1, h2, h3, h4, h5, h6, p, div {
            color: #ffffff !important;
        }
        .main-title {
            text-align: center;
            color: #ffffff;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            padding-top: 30px;
        }
        div.stButton > button {
            font-size: 28px !important;
            font-weight: bold;
            padding: 18px;
            border-radius: 12px;
            border: none;
            color: white;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            transition: all 0.3s ease;
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
            width: 300px;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        div.stDownloadButton > button {
            font-size: 18px !important;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 12px;
            border: none;
            color: white !important;
            background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
            transition: all 0.3s ease;
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        }
        div.stDownloadButton > button:hover {
            background: linear-gradient(135deg, #2575fc, #6a11cb) !important;
            transform: scale(1.05);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #000000, #1a1a1a) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            font-size: 28px !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label span {
            font-size: 28px !important;
        }
        [data-testid="stSidebar"] .stRadio {
            font-size: 28px !important;
        }
        [data-testid="stSidebar"] .stRadio > div > label > div > p {
            font-size: 28px !important;
        }
        .stMarkdown h3 {
            color: #ffffff !important;
            font-size: 24px !important;
        }
        .stFileUploader {
            background-color: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        .stFileUploader label {
            color: #ffffff !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background-color: #1a1a1a !important;
            border: 2px dashed #333333 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stFileUploaderDropzone"] div {
            color: #ffffff !important;
        }
        .stFileUploader div[role="button"] {
            background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
        }
        .stFileUploader div[role="button"]:hover {
            background: linear-gradient(135deg, #2575fc, #6a11cb) !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'gif_key' not in st.session_state:
    st.session_state.gif_key = 0
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'destination' not in st.session_state:
    st.session_state.destination = None
if 'recognized_results' not in st.session_state:
    st.session_state.recognized_results = []
if 'last_choice' not in st.session_state:
    st.session_state.last_choice = None  # 이전 메뉴 상태 저장


def main():
    # 메뉴 설정
    # menu = ["홈", "비디오 파일 분석", "키포인트 파일 분석", "실시간 웹캠 인식", "키오스크 모드"]
    # choice = st.sidebar.radio("메뉴 선택", menu)

    with st.sidebar:
        choice = option_menu(
            "메뉴 선택", 
            ["홈", "비디오 파일 분석", "키포인트 파일 분석", "실시간 웹캠 인식", "키오스크 모드"],
            icons=['house', 'film', 'share', 'webcam', 'display'],
            menu_icon="app-indicator", 
            default_index=0,
            styles={
                "container": {"padding": "4!important", "background-color": "black"},
                "menu-title": {  # "메뉴 선택" 부분 스타일 지정
                    "font-size": "26px",
                    "font-weight": "bold",
                    "color": "#cccccc",  # 진한 흰색으로 변경
                    "text-align": "left",
                },
                "icon": {"color": "white", "font-size": "25px"},
                "nav-link": {
                    "font-size": "24px", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "color": "#cccccc",  # 기존 메뉴 글씨 유지
                    "--hover-color": "#333333"
                },
                "nav-link-selected": {"background-color": "#4843E3"},
            }
        )


    if choice == "홈":
        home.app()
    elif choice == "비디오 파일 분석":
        video.app()
    elif choice == "키포인트 파일 분석":
        keypoint.app()
    elif choice == "실시간 웹캠 인식":
        webcam.app()
    elif choice == "키오스크 모드":
        kiosk.main()

if __name__ == "__main__":
    main()