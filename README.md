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

### 1. 데이터셋 준비

이 레포는 **FaceForensics++ (FF++) C23** 데이터셋을 사용합니다.  
데이터는 라이선스 문제로 Git에 포함되어 있지 않으며, 각자 로컬에서 직접 받아야 합니다.

1. **Kaggle에서 FF++ C23 다운로드**
   - Kaggle 데이터셋: https://www.kaggle.com/datasets/xdxd003/ff-c23  
   - Kaggle 계정으로 로그인한 뒤, "Download" 버튼을 눌러 전체 데이터를 받습니다.  
2. 압축을 해제한 뒤, 이 레포 기준으로 아래와 같은 경로가 되도록 맞춰줍니다:

```
TruthLens-Model/
├── data/
│   ├── raw/
|       ├── ffpp_c23/
|           ├── FaceForensics++_C23/
|               ├── original/
|               ├── Deepfakes/
|               ├── Face2Face/
|               ├── FaceSwap/
|               ├── NeuralTextures/
|                ...
```
3. `data/` 폴더는 `.gitignore`에 의해 Git에 올라가지 않으므로,  
**모든 팀원은 위 경로 구조를 로컬에서 동일하게 맞춰야 합니다.**

> 이후 단계(프레임 추출, 얼굴 추출, 인덱스 생성 등)는  
> `src/*.py` 스크립트가 필요한 디렉토리를 자동으로 생성합니다.  
> 직접 만들 필요 없이 1번 단계(데이터 위치)만 맞으면 됩니다.

### 2. 환경 설정

```
# (선택) 가상환경 생성
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 기본 라이브러리 설치
pip install -r requirements.txt
```

### 3. 모델 파일 다운로드
훈련된 모델 가중치는 weights/ 디렉토리에 저장됩니다.
(.pth 파일은 Git LFS로 관리)

---

## 🔄 전체 처리 파이프라인
TruthLens 모델은 다음 순서로 실행됩니다.

1) FF++ 원본 영상 → 프레임 추출
```
python -m src.extract_frames_ffpp
```
2) 프레임 → 얼굴 crop 추출
```
python -m src.extract_faces_ffpp
```
3) train/val/test split 생성
```
python -m src.build_ffpp_index
```
4) Baseline 모델 학습 (ResNet18)
```
python -m src.train_ffpp_baseline

출력:
weights/ffpp_resnet18_baseline.pth
```
5) Improved 모델 학습 (ResNet50)
```
python -m src.train_ffpp_resnet50

출력:
weights/ffpp_resnet50_advanced.pth
```
6) 테스트셋 평가
```
# Baseline:
python -m src.evaluate_ffpp_baseline
```

```
# Improved:
python -m src.evaluate_ffpp_resnet50
```

---

## 📂 프로젝트 구조

```
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
```

---