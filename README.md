**TruthLens**는 이미지 및 비디오 프레임 내에서 얼굴 영역을 자동으로 추출하고, 두 개의 고도화된 딥러닝 백본 네트워크를 가중 앙상블(Weighted Ensemble)하여 딥페이크 여부를 정밀하게 판별하는 인공지능 탐지 엔진입니다. 

실제 웹 환경에서 발생하는 고압축 손실 및 저화질 환경에 강인하도록 데이터 증강 및 최적화 전략을 반영하였습니다.

---

## 📌 Key Features

- **Automated Face Targeting**: `MTCNN` 기반의 실시간 얼굴 탐지 및 크롭을 수행하여 배경 노이즈를 완전히 제거하고 얼굴 관심 영역(ROI)에만 집중합니다.
- **Hybrid Ensemble Logic**: `EfficientNet-B0`와 `MobileNet-V3 Large` 모델을 **8:2 비율로 결합한 Soft Voting 앙상블**을 적용하여 탐지 안정성과 연산 효율성을 동시에 극대화했습니다.
- **Robust Generalization**: 일반적인 Cross-Entropy 대신 `Focal Loss`를 적용하여, 구별하기 어려운 하드 샘플(Hard Samples)에 대한 오차 보정 능력을 대폭 끌어올렸습니다.

---


## 📂 Project Directory Structure

```text
project_root/
│
├── model/                     # 📂 AI Model Definition & Inference Interface
│   ├── __init__.py            # Package interface initialization
│   ├── config.py              # Model metadata and weights path configurations
│   ├── inference.py           # End-to-end unified inference API bridge
│   ├── model.py               # Core backbone architectures (EffB0 / MobV3 Builder)
│   └── preprocess.py          # Standardized ImageNet pre-processing utilities
│
├── src/                       # 📂 Core Execution Pipelines & Data Components
│   ├── build_index.py         # Video-to-frame extraction & 8:1:1 split manager
│   ├── dataset.py             # Robust data augmentation & PyTorch Dataset loader
│   ├── evaluate_ensemble.py   # Soft voting ensemble metrics evaluation dashboard
│   ├── paths.py               # Environment-agnostic dynamic path resolver
│   └── train_ensemble.py      # Core optimization engine with Focal Loss
│
├── weights/                   # 📂 Pre-trained Checkpoint Artifacts (.pth storage)
│   ├── dfdc_efficientnet_b0_focal.pth
│   └── dfdc_mobilenet_v3_focal.pth
│
├── README.md                  # Project documentation registry
└── requirements.txt           # Managed system dependency specifications

```

---

## 📊 Datasets

본 프로젝트는 모델의 범용성과 위조 경계면 분별력을 극대화하기 위해 아래 두 가지 글로벌 표준 벤치마크 데이터셋을 결합하여 가공 및 학습을 진행하였습니다.

1. **Celeb-DF (v2)**
* **Description**: 고품질 연예인 유튜브 영상을 기반으로 생성된 데이터셋으로, 미세한 얼굴 노이즈 및 정밀한 변조 텍스처를 학습하는 데 활용되었습니다.
* **Link**: [Celeb-DF GitHub Repository](https://github.com/yuezunli/celeb-deepfakeforensics)


2. **DFDC (Deepfake Detection Challenge)**
* **Description**: Meta(Facebook) 주도의 대규모 데이터셋으로, 실제 소셜 미디어 웹 환경에서 발생하는 다양한 조명 변화, 저화질 열화, 고압축 환경에 대한 강인성(Robustness)을 확보하기 위해 활용되었습니다.
* **Link**: [DFDC Official Dataset Page (Kaggle)](https://www.kaggle.com/c/deepfake-detection-challenge)



---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/TruthLens-Model.git](https://github.com/your-username/TruthLens-Model.git)
cd TruthLens-Model

# Install managed dependencies
pip install -r requirements.txt

```

---

## 🚀 Core Learning Pipeline

### 1. Data Preprocessing (`src/build_index.py` & `src/dataset.py`)

* **Dataset Hybridization**: 고품질 연예인 데이터셋인 `Celeb-DF`와 실제 압축 환경을 모사한 `DFDC(Deepfake Detection Challenge)` 데이터를 융합하여 도메인 범용성을 확보했습니다.
* **Advanced Augmentation**: 미세한 얼굴 각도 변화에 대응하기 위한 `RandomRotation(15)`, 조명 편차 극복을 위한 `ColorJitter`, 그리고 고해상도 위조 흔적에 무뎌지지 않고 미세 텍스처를 꼼꼼하게 찾도록 유도하는 `RandomErasing(p=0.3)`을 배치했습니다.

### 2. Model Architecture (`model/model.py` & `model/preprocess.py`)

* **Primary Backbone (`EfficientNet-B0`)**: 고차원 특징 추출 능력이 뛰어나 위조 경계면 파악에 우수한 성능을 보입니다.
* **Lightweight Backbone (`MobileNet-V3`)**: 가벼우면서도 빠른 연산 속도를 보장하여, 앙상블 시 추론 속도 저하를 방지하고 탐지 결과의 편차를 줄여주는 서포터 역할을 수행합니다.
* **Preprocessing**: `model/preprocess.py`를 통해 ImageNet 표준 정규화 가중치를 적용하여 백본 입력 텐서의 정밀도를 통일합니다.

### 3. Optimization Strategy (`src/train_ensemble.py`)

* **Loss Function**: `Focal Loss ($\gamma=2.0$, $\alpha=1.0$)`를 이식하여, 이미 잘 맞추는 쉬운 샘플의 가중치는 낮추고 탐지가 까다로운 고압축 위조 영상에 가중치를 부여해 최적화를 가속했습니다.
* **Learning Rate Scheduler**: `CosineAnnealingLR`을 결합하여 가중치 수렴의 안정성을 확보했습니다.

---

## 🎯 Evaluation Dashboard Summary (`src/evaluate_ensemble.py`)

본 프로젝트는 테스트 데이터셋을 기반으로 단독 모델과 최종 앙상블 모델의 성능을 비교 검증할 수 있는 통합 대시보드를 제공합니다.

```bash
python -m src.evaluate_ensemble --trained-domain dfdc --target-domain dfdc --batch-size 64

```

### Expected Output Format

```text
============================================================
📊 Final Ensemble Evaluation Dashboard Summary
============================================================
[1. EfficientNet-B0 Backbone (Standalone)]
  - Accuracy  : 92.15%
  - ROC-AUC   : 0.9486
  - Precision : 91.20%
  - Recall    : 93.10%
  - F1-Score  : 92.14%

[2. MobileNet-V3 Backbone (Standalone)]
  - Accuracy  : 87.40%
  - ROC-AUC   : 0.8924
  - Precision : 86.50%
  - Recall    : 88.30%
  - F1-Score  : 87.39%

[🌟 3. Integrated Production Hybrid Ensemble (Final Outcome)]
  - Accuracy  : 94.65%
  - ROC-AUC   : 0.9652
  - Precision : 93.80%
  - Recall    : 95.50%
  - F1-Score  : 94.64%

```

*Soft Voting 앙상블 기법을 적용했을 때 단독 모델 대비 평가지표 전반(Accuracy, AUC)이 크게 향상되는 시너지 효과를 검증했습니다.*

---

## 💻 Backend Integration Guide (`model/inference.py`)

백엔드 엔지니어 및 타 파트 개발자는 내부 아키텍처나 전처리 과정을 깊게 파고들 필요 없이, `DeepfakeDetector` 인터페이스 클래스 단 **3줄 호출**만으로 정밀한 딥페이크 판별 기능을 서비스 시스템에 연동할 수 있습니다.

```python
from model.inference import DeepfakeDetector

# 1. Initialize the detector (Automatically detects and allocates CUDA/GPU if available)
detector = DeepfakeDetector()

# 2. Run inference pipeline (Executes MTCNN face crop -> Ensemble predicting)
result = detector.predict("path/to/target_user_image.jpg")

# 3. Utilize the structural JSON response
if result["status"] == "success":
    print(f"Prediction Result : {result['label']}")  # REAL or FAKE
    print(f"Confidence Score  : {result['score']}%")  # Fake probability (0.0% ~ 100.0%)

```

### API Response Format (JSON)

```json
{
  "status": "success",
  "label": "FAKE",
  "score": 94.65,
  "details": {
    "efficientnet_b0": 0.9550,
    "mobilenet_v3": 0.9125
  }
}

```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
