# TruthLens-Model
딥러닝 기반 딥페이크·이미지 위변조 탐지 서비스 **TruthLens**의  
**모델 개발 전용 저장소**입니다.

FastAPI 백엔드와 분리된 독립적인 머신러닝 파이프라인을 구성하고 있으며,  
FF++(FaceForensics++) 기반으로 학습된 ResNet18/ResNet50 모델을 제공합니다.

---

## 🧠 사용된 기술 스택
- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Models**:
  - ResNet18 (Baseline)
  - ResNet50 + Label Smoothing + AdamW + Cosine LR (Improved)
- **Dataset**: FaceForensics++ (FF++ C23)

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# (선택) 가상환경 생성
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 기본 라이브러리 설치
pip install -r requirements.txt
2. 모델 파일 다운로드
훈련된 모델 가중치는 weights/ 디렉토리에 저장됩니다.
(.pth 파일은 Git LFS로 관리)

🔄 전체 처리 파이프라인
TruthLens 모델은 다음 순서로 실행됩니다.

1) FF++ 원본 영상 → 프레임 추출
bash
코드 복사
python -m src.extract_frames_ffpp
2) 프레임 → 얼굴 crop 추출
bash
코드 복사
python -m src.extract_faces_ffpp
3) train/val/test split 생성
bash
코드 복사
python -m src.build_ffpp_index
4) Baseline 모델 학습 (ResNet18)
bash
코드 복사
python -m src.train_ffpp_baseline
출력:
weights/ffpp_resnet18_baseline.pth

5) Improved 모델 학습 (ResNet50)
bash
코드 복사
python -m src.train_ffpp_resnet50
출력:
weights/ffpp_resnet50_advanced.pth

6) 테스트셋 평가
Baseline:

bash
코드 복사
python -m src.evaluate_ffpp_baseline
Improved:

bash
코드 복사
python -m src.evaluate_ffpp_resnet50
📂 프로젝트 구조
powershell
코드 복사
TruthLens-Model/
├── src/
│   ├── extract_frames_ffpp.py       # 영상 → 프레임 추출
│   ├── extract_faces_ffpp.py        # 프레임 → 얼굴 crop
│   ├── build_ffpp_index.py          # train/val/test CSV 생성
│   ├── dataset_ffpp.py              # Dataset 정의
│   ├── train_ffpp_baseline.py       # ResNet18 학습
│   ├── train_ffpp_resnet50.py       # ResNet50 학습
│   ├── evaluate_ffpp_baseline.py    # ResNet18 평가
│   ├── evaluate_ffpp_resnet50.py    # ResNet50 평가
│   └── paths.py                     # 공통 경로 관리
│
├── weights/                         # 학습된 모델 (.pth, LFS)
├── data/                            # 로컬 데이터 (Git 제외)
├── requirements.txt
└── README.md
📝 주요 기능 요약
FF++ 데이터셋 기반 영상/이미지 전처리 자동화

얼굴 detection → crop → resize(224x224)

Baseline / Improved 모델 학습 가능

train/val/test split 자동 생성

모델 평가 및 정확도 출력

백엔드에서 사용할 .pth 가중치 생성

✨ 추후 TODO
영상 단위 예측 aggregation (frame → video-level score)

Grad-CAM 시각화

EfficientNet, ViT 등 추가 모델 실험

ONNX 변환 & FastAPI 연동 최적화
