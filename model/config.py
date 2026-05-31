# model/config.py

from pathlib import Path
from typing import Any, Dict

# Base Directory Setup
# BASE_DIR: .../TruthLens-Model/model
BASE_DIR = Path(__file__).resolve().parent
# WEIGHTS_DIR: .../TruthLens-Model/weights
WEIGHTS_DIR = BASE_DIR.parent / "weights"

MODEL_LIST: Dict[str, Dict[str, Any]] = {
    # 팀원 원본 DFDC 학습 모델
    "efficientnet_b0_dfdc": {
        "weights": WEIGHTS_DIR / "dfdc_efficientnet_b0_focal.pth",
        "arch": "efficientnet_b0",
        "num_classes": 2,
        "threshold": 0.5,
    },
    "mobilenet_v3_dfdc": {
        "weights": WEIGHTS_DIR / "dfdc_mobilenet_v3_focal.pth",
        "arch": "mobilenet_v3",
        "num_classes": 2,
        "threshold": 0.5,
    },
    # 파인튜닝 EfficientNet-B0 (Celeb-DF + DFDC, AUC 0.8016)
    "eff_b0_finetuned": {
        "weights": BASE_DIR / "weights" / "eff_b0_finetuned.pth",
        "arch": "efficientnet_b0",
        "num_classes": 2,
        "threshold": 0.5,
    },
}

DEFAULT_MODEL_NAME = "eff_b0_finetuned"

# 최종 앙상블: 파인튜닝 EfficientNet(×0.8) + DFDC MobileNet(×0.2)
ENSEMBLE_CONFIG: Dict[str, Any] = {
    "models": [
        {"name": "eff_b0_finetuned",  "weight": 0.8},
        {"name": "mobilenet_v3_dfdc", "weight": 0.2},
    ],
    "threshold": 0.5,
}