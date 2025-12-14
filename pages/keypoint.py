import streamlit as st
import base64
import os
from utils.sign_app import tempfile, predict_sign_language, load_model_and_mapping
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

    st.subheader("키포인트 파일 분석")
    uploaded_file = st.file_uploader("샘플 키포인트 파일(.txt)을 업로드하세요", type=["txt"])
    
    if uploaded_file is not None:
        st.write(f"업로드된 파일: {uploaded_file.name}")
        try:
            txt_content = uploaded_file.getvalue().decode('utf-8')
        except UnicodeDecodeError as e:
            st.error(f"파일 디코딩 오류: {e}")
            st.write("파일이 UTF-8 인코딩이 아닌 것 같습니다. UTF-8로 인코딩된 파일을 업로드해주세요.")
            return
        
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

if __name__ == "__main__":
    app()