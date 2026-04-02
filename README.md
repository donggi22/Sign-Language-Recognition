디렉토리명을 pages -> page로 변경 (pages로 할 경우 streamlit 예약 명령어로 사이드바 자꾸 생성됨)

# Streamlit 웹 애플리케이션 실행 가이드

## 가상환경 생성
## MediaPipe 호환성 (Python 3.8 ~ 3.11 ver 지원)
conda create -n sign python=3.11 -y

## 가상환경 활성화
conda activate sign

## 프로젝트 디렉토리로 이동
cd Sign-Language-Recognition

## 필요한 패키지 설치
pip install -r requirements.txt

## stremalit 웹 애플리케이션 실행
streamlit run app.py
