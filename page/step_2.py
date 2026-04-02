import streamlit as st
from PIL import Image
import base64
from utils.func import gif_paths, get_base64_of_file, go_to_step, model_path, mapping_path
import cv2
import time
from utils import sign_app as sa

# 변수 초기화
last_recognition_time = 0
MIN_TIME_BETWEEN_RECOGNITIONS = 2.0

def app():
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔙뒤로"):
            go_to_step(0)

    try:
        model, id_to_label, device = sa.load_model_and_mapping(model_path, mapping_path)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        model = None
        id_to_label = {}
    try:
        gif_path = gif_paths[1]
        data_url = get_base64_of_file(gif_path)
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" style="display: block; margin: 0 auto;" alt="Welcome GIF">',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"GIF 파일을 찾을 수 없습니다: {gif_paths[1]}")
    st.markdown("<h2 style='text-align: center; color: #333;'>웹캠시작을 누른 후<br>궁금하신 장소를 수어로 표현해주세요.</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.75, 1, 1])
    with col2:
        if not st.session_state.webcam_active:
            if st.button("📸웹캠 시작"):
                st.session_state.webcam_active = True
                st.rerun()
        else:
            if st.button("⏸️인식 중지", key="stop_button"):
                st.session_state.webcam_active = False
                st.rerun()

    # # 캠 넘어가기용 최종적으로 제거예정정
    # col1, col2, col3 = st.columns([0.75, 1, 1])
    # with col2:
    #     if st.button("🔜다음"):
    #         go_to_step(6)

    if st.session_state.webcam_active:
        with st.container():
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                video_frame = st.empty()
                # 로딩 문구 추가
                loading_placeholder = st.empty()
                loading_placeholder.markdown(
                    "<h3 style='text-align: center; color: #333;'>웹캠을 불러오는<br> 중입니다...</h3>",
                    unsafe_allow_html=True
                )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            result_text = st.empty()
            debug_text = st.empty()

        recognizer = sa.RealTimeSignRecognizer(
            model=model,
            id_to_label=id_to_label,
            buffer_size=150,
            movement_threshold=0.05,
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

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("웹캠을 열 수 없습니다.")
            st.session_state.webcam_active = False
            loading_placeholder.empty()  # 오류 시 로딩 문구 제거
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # 웹캠이 성공적으로 열리면 로딩 문구 제거
            loading_placeholder.empty()

            frame_count = 0
            global last_recognition_time
            last_recognition_time = 0

            while cap.isOpened() and st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("웹캠 프레임 읽기 실패")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                keypoints_data = sa.convert_mediapipe_to_openpose(results, frame.shape)
                annotated_image = sa.draw_openpose_style_landmarks(frame_rgb, keypoints_data)

                result, confidence, top3_pred, is_signing = recognizer.process_frame(frame, results)

                video_frame.image(annotated_image, channels='RGB', width=960)

                current_time = time.time()

                if top3_pred:
                    top_result, top_confidence = top3_pred[0]
                    if (top_result not in st.session_state.recognized_results or 
                        current_time - last_recognition_time >= MIN_TIME_BETWEEN_RECOGNITIONS):
                        try:
                            gif_path = gif_paths[1]
                            data_url = get_base64_of_file(gif_path)
                            html_content = (
                                f"<h3 style='text-align: center; color: #333;'>인식된 목적지 : {top_result}<br></h3>"
                                f'<img src="data:image/gif;base64,{data_url}" style="display: block; margin: 0 auto;" alt="Welcome GIF"><br>'
                                f"<h3 style='text-align: center; color: #333;'>목적지를 선택하려면 인식중지를 눌러주세요.</h3>"
                            )
                            result_text.markdown(html_content, unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.error(f"GIF 파일을 찾을 수 없습니다: {gif_paths[1]}")
                        if top_result not in st.session_state.recognized_results:
                            st.session_state.recognized_results.append(top_result)
                            last_recognition_time = current_time

                frame_count += 1
                time.sleep(0.01)

            if cap.isOpened():
                cap.release()
            holistic.close()
            st.rerun()

    if not st.session_state.webcam_active and st.session_state.recognized_results:
        st.markdown("<h3 style='text-align: center; color: #333;'>인식된 목적지 목록:</h3>", unsafe_allow_html=True)
        
        for idx, result in enumerate(st.session_state.recognized_results):
            col1, col2, col3 = st.columns([0.75, 1, 1])
            with col2:
                if st.button(f"🔜'{result}'(으)로 진행", key=f"proceed_{result}_{idx}"):
                    st.session_state.destination = result
                    if result == '경복궁':
                        go_to_step(3)
                    
                    elif result == '청와대':
                        go_to_step(6)

                    else:
                        st.session_state.warning_message = "⚠️ 이 장소는 현재 준비 중입니다. 조금만 기다려 주세요!"
                        st.markdown(f"<p style='text-align: center; color: red;'>{st.session_state.warning_message}</p>", unsafe_allow_html=True)