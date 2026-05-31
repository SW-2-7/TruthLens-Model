# model/pipeline.py
"""
백엔드 팀이 사용할 최종 딥페이크 탐지 파이프라인.
앙상블: EfficientNet-B0 파인튜닝(×0.8) + MobileNetV3-DFDC(×0.2)
AUC: 0.8016 (DeepFake-Eval-2024)

사용법:
    from model.pipeline import DeepfakeDetectionPipeline

    pipeline = DeepfakeDetectionPipeline()   # 서버 시작 시 1회만
    result   = pipeline.run(pil_image)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .config import ENSEMBLE_CONFIG, MODEL_LIST
from .model import create_model
from .preprocess import preprocess_pil


# ── 모델 로딩 유틸 ────────────────────────────────────────────────────────────

def _load_model(model_name: str, device: str) -> nn.Module:
    """MODEL_LIST 설정으로 단일 모델을 로드해 eval 상태로 반환."""
    cfg = MODEL_LIST[model_name]
    weights_path: Path = cfg["weights"]

    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일 없음: {weights_path}")

    model = create_model(arch=cfg["arch"], num_classes=cfg["num_classes"])
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    # state_dict 추출
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # DataParallel 'module.' prefix 제거
    if isinstance(ckpt, dict) and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

    model.load_state_dict(ckpt)
    return model.to(device).eval()


# ── 파이프라인 ────────────────────────────────────────────────────────────────

class DeepfakeDetectionPipeline:
    """
    PIL 이미지 1장을 받아 딥페이크 여부를 반환하는 앙상블 파이프라인.

    반환값 예시
    ----------
    성공 시::

        {
            "success": True,
            "label": "FAKE",
            "fake_probability": 0.91,
            "real_probability": 0.09,
            "confidence": "high",
            "confidence_score": 0.82,
        }

    얼굴 탐지 실패 시::

        {
            "success": False,
            "error": "no_face",
            "message": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 사진을 올려주세요.",
        }
    """

    def __init__(
        self,
        use_face_crop: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 딥페이크 탐지 파이프라인 초기화 (device={self.device})")

        # 앙상블 모델 로드
        self._models: List[Dict[str, Any]] = []
        for cfg in ENSEMBLE_CONFIG["models"]:
            model = _load_model(cfg["name"], self.device)
            self._models.append({"model": model, "name": cfg["name"], "weight": cfg["weight"]})
            print(f"[INFO]   ✓ {cfg['name']}  (weight={cfg['weight']})")

        self.threshold: float = float(ENSEMBLE_CONFIG["threshold"])
        self.use_face_crop = use_face_crop

        if use_face_crop:
            self._init_face_detector()

        print("[INFO] 파이프라인 준비 완료")

    def _init_face_detector(self) -> None:
        try:
            from .face_detectors import make_detector
            self.face_detector = make_detector("mtcnn")
            print("[INFO] 얼굴 탐지기(MTCNN) 로드 완료")
        except Exception as e:
            print(f"[WARN] 얼굴 탐지기 로드 실패, 전체 이미지로 진행합니다: {e}")
            self.use_face_crop = False

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def run(self, img: Image.Image) -> Dict[str, Any]:
        """PIL 이미지를 받아 딥페이크 판별 결과를 반환."""
        img = img.convert("RGB")

        if self.use_face_crop:
            cropped = self._crop_face(img)
            if cropped is None:
                return {
                    "success": False,
                    "error": "no_face",
                    "message": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 사진을 올려주세요.",
                }
            img = cropped

        try:
            fake_prob = self._ensemble_score(img)
        except Exception as e:
            return {
                "success": False,
                "error": "model_error",
                "message": f"모델 추론 오류: {e}",
            }

        real_prob = round(1.0 - fake_prob, 4)
        confidence_score = round(abs(fake_prob - 0.5) * 2, 4)

        return {
            "success": True,
            "label": "FAKE" if fake_prob >= self.threshold else "REAL",
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": self._confidence_level(confidence_score),
            "confidence_score": confidence_score,
        }

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ensemble_score(self, img: Image.Image) -> float:
        """앙상블 가중 평균으로 fake 확률 반환."""
        x = preprocess_pil(img).to(self.device)
        score = 0.0
        for item in self._models:
            prob = torch.softmax(item["model"](x), dim=1)[0, 1].item()
            score += item["weight"] * prob
        return round(score, 4)

    def _crop_face(self, img: Image.Image) -> Optional[Image.Image]:
        """MTCNN으로 얼굴만 crop. 실패 시 None."""
        try:
            import cv2
            import numpy as np
            from .face_detectors import crop_face
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detection = self.face_detector.detect_main_face(img_bgr)
            if detection is None:
                return None
            face_bgr = crop_face(img_bgr, detection, output_size=224)
            if face_bgr is None or face_bgr.size == 0:
                return None
            return Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            return None

    @staticmethod
    def _confidence_level(score: float) -> str:
        if score < 0.2:
            return "low"
        if score < 0.5:
            return "medium"
        return "high"
