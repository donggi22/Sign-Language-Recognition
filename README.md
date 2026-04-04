# 대중교통 이용 상황에서의 청각장애 농인을 위한 양방향 수어 통역 시스템

## 프로젝트 구조
```
핵심 구조:
├── app.py                    # Streamlit 메인 앱
├── page/                    # 6단계 키오스크 플로우
├── utils/                    # TCN 모델 + 실시간 인식 엔진
├── models/                   # 10.2MB 사전 학습 모델 (995 클래스)
├── gif_images/               # 6개 안내 애니메이션
├── 데이터 구축 및 학습 코드    # 데이터 전처리 및 ver별 모델 code
└── 샘플 데이터/               # 테스트용 비디오/키포인트 파일
```

디렉토리명을 pages -> page로 변경 (pages로 할 경우 streamlit 예약 명령어로 사이드바 생성됨)

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
