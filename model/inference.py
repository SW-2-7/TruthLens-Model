# model/inference.py

from typing import Dict, Any, Optional

import torch
from PIL import Image

from .config import MODEL_LIST, DEFAULT_MODEL_NAME
from .model import create_model
from .preprocess import preprocess_pil


def _clean_state_dict(state: dict) -> dict:
    """
    DataParallel로 학습해서 key 앞에 'module.' 붙어 있는 경우 제거
    """
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    모델과 가중치를 로딩해서 eval 상태로 반환.

    - model_name: config.MODEL_LIST 키 값
    - device: "cpu", "cuda" 등. None이면 자동 선택.
    """
    if model_name not in MODEL_LIST:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: {list(MODEL_LIST.keys())}"
        )

    # device 자동 선택
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MODEL_LIST[model_name]

    # 아키텍처 생성
    model = create_model(
        arch=cfg["arch"],
        num_classes=cfg["num_classes"],
    )

    # 가중치 로딩
    weights_path = cfg["weights"]
    print(f"[INFO] Loading model: {weights_path}")
    state = torch.load(weights_path, map_location=device)

    # 만약 저장 형식이 {"state_dict": ...}라면 처리
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = _clean_state_dict(state)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    # config 정보에서 threshold / 모델 이름을 모델에 심어두기
    setattr(model, "threshold", float(cfg.get("threshold", 0.5)))
    setattr(model, "model_name", model_name)

    return model


def predict_from_path(
    model: torch.nn.Module,
    image_path: str,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    이미지 경로를 받아서 바로 예측하는 helper 함수.
    """
    img = Image.open(image_path).convert("RGB")
    return predict_from_pil(
        model=model,
        img=img,
        device=device,
        threshold=threshold,
    )


@torch.no_grad()
def predict_from_pil(
    model: torch.nn.Module,
    img: Image.Image,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    PIL 이미지 한 장을 받아서 fake 확률과 label 리턴.

    클래스 인덱스 가정:
    - class 0: REAL
    - class 1: FAKE
    """
    # device 처리
    if device is None:
        # 이미 model이 올라가 있는 device를 따라감
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)

    # threshold가 None이면 모델에 심어둔 값 사용, 없으면 0.5
    if threshold is None:
        threshold = float(getattr(model, "threshold", 0.5))

    x = preprocess_pil(img).to(device)  # (1, 3, H, W)
    logits = model(x)                   # (1, num_classes)

    probs = torch.softmax(logits, dim=1)[0]
    real_prob = float(probs[0].item())
    fake_prob = float(probs[1].item())

    label = "FAKE" if fake_prob >= threshold else "REAL"

    return {
        "label": label,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "threshold": threshold,
        "model_name": getattr(model, "model_name", None),
    }
