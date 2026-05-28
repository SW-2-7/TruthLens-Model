# model/__init__.py

from .config import DEFAULT_MODEL_NAME, MODEL_LIST
from .inference import DeepfakeDetector
from .model import create_model

# 외부 노출 인터페이스 정의
__all__ = [
    "MODEL_LIST",
    "DEFAULT_MODEL_NAME",
    "create_model",
    "DeepfakeDetector",
]