# model/config.py

from pathlib import Path
from typing import Dict, Any

# 이 파일이 위치한 폴더(model/) 기준 경로
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

# 나중에 모델이 늘어나면 이 dict만 수정/추가하면 됨
MODEL_LIST: Dict[str, Dict[str, Any]] = {
    "resnet18_ffpp": {
        "weights": WEIGHTS_DIR / "ffpp_resnet18.pth",
        "arch": "resnet18",
        "num_classes": 2,   # 0: REAL, 1: FAKE 가정
        "threshold": 0.5,
    },

    "resnet50_ffpp": {
        "weights": WEIGHTS_DIR / "ffpp_resnet50.pth",
        "arch": "resnet50",
        "num_classes": 2,
        "threshold": 0.5,
    },

    # ✅ fine-tuned 모델 추가
    "resnet50_celebdf": {
        "weights": WEIGHTS_DIR / "celebdf_resnet50.pth",
        "arch": "resnet50",
        "num_classes": 2,
        "threshold": 0.5,   # 필요하면 나중에 모델별로 다르게 설정 가능
    },
}

# ✅ 기본으로 쓸 모델을 fine-tuned 으로 변경
DEFAULT_MODEL_NAME = "resnet50_celebdf"
