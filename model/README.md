# TruthLens Model Package

이 폴더는 딥페이크 / 이미지 위변조 탐지용 **모델 패키지**입니다.  
백엔드는 이 디렉토리만 레포에 추가해서 바로 사용할 수 있습니다.

---

## 📁 구성

- `__init__.py`  
  - 공개 API 정리 (`load_model`, `predict_from_pil`, `predict_from_path`, `MODEL_LIST`, `DEFAULT_MODEL_NAME`)
- `config.py`  
  - 사용 가능한 모델 목록, 가중치 경로 정의
- `model.py`  
  - ResNet18 / ResNet50 기반 분류 모델 아키텍처 정의
- `preprocess.py`  
  - FF++ 학습 시 사용한 **검증/테스트용 전처리**와 동일한 transform
- `inference.py`  
  - 모델 로딩 및 추론 함수 구현
- `weights/`  
  - 학습된 PyTorch `state_dict` (`.pth`) 파일 저장

---

## 🔧 설치 / 사용 방법

이 폴더(`model/`)를 백엔드 레포에 그대로 복사한 후 필요한 라이브러리 설치를 진행합니다.<br />
필요한 라이브러리는 TRUTHLENS-MODEL 레포지토리 내부에 있는 `requirements_inference.txt`를 사용합니다.
```python
pip install -r requirements_inference.txt
```

백엔드 코드에서 다음과 같이 import 해서 사용합니다.

```python
from model import load_model, predict_from_pil, DEFAULT_MODEL_NAME
from PIL import Image

# 1) 모델 로드
device = "cuda"  # or "cpu"
model = load_model(DEFAULT_MODEL_NAME, device=device)

# 2) PIL.Image 기반 추론
img = Image.open("some_image.jpg")
result = predict_from_pil(model, img, device=device)
print(result)
# -> {
#      "label": "FAKE" or "REAL",
#      "fake_probability": float,
#      "real_probability": float,
#      "threshold": 0.5,
#    }

# 3) 파일 경로 기반 추론 helper
from model import predict_from_path

result2 = predict_from_path(model, "some_image.jpg", device=device)
print(result2)
```

## 📌 지원 모델 목록

`config.py`의 `MODEL_LIST`에 정의되어 있습니다.

현재:

- `resnet50_ffpp`

   - FF++ 기반 ResNet50 improved 모델

- `resnet18_ffpp`

  - FF++ 기반 ResNet18 baseline 모델

기본 모델 이름은 `DEFAULT_MODEL_NAME`로 관리하며,<br />
현재 값은:

```
DEFAULT_MODEL_NAME = "resnet50_ffpp"
```


입니다.

## 🔒 주의 사항

이 모델은 FF++ 및 유사한 데이터 분포를 대상으로 학습되었습니다.

일반적인 모든 사진/영상에 대해 완전한 진위 판별을 보장하지 않습니다.

새로운 데이터셋으로 학습된 모델이 추가되면,
weights/에 .pth 파일을 추가하고 config.py의 MODEL_LIST에 항목만 추가하면 됩니다.

---