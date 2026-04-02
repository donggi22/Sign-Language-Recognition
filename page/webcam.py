import streamlit as st
from utils import sign_app as sa
from utils.sign_app import process_webcam, load_model_and_mapping
from utils.func import model_path, mapping_path

def app():
    try:
        model, id_to_label, device = load_model_and_mapping(model_path, mapping_path)
        st.sidebar.success("모델 로드 완료!")
    except Exception as e:
        st.sidebar.error(f"모델 로드 중 오류 발생: {e}")
        return
    
    # 세션 상태 초기화
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

        recognizer = sa.RealTimeSignRecognizer(
            model=model,
            id_to_label=id_to_label,
            buffer_size=150,
            movement_threshold=0.1,
            idle_frames=15,
            confidence_threshold=0.01,
            face_weight=3.0
        )
        holistic = sa.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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