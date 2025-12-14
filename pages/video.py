import streamlit as st
import base64
from utils.sign_app import extract_keypoints_from_video, render_debug_images, predict_sign_language, load_model_and_mapping
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