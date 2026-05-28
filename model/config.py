# model/config.py

from pathlib import Path
from typing import Any, Dict

# Base Directory Setup
# BASE_DIR: .../TruthLens-Model/model
BASE_DIR = Path(__file__).resolve().parent
# WEIGHTS_DIR: .../TruthLens-Model/weights
WEIGHTS_DIR = BASE_DIR.parent / "weights"

# 🚀 Production-ready Ensemble Model Configurations
# EfficientNet-B0 (Weight: 0.8) + MobileNet-V3 (Weight: 0.2)
MODEL_LIST: Dict[str, Dict[str, Any]] = {
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
    }
}

# Default baseline model selection for single inference fallback
DEFAULT_MODEL_NAME = "efficientnet_b0_dfdc"