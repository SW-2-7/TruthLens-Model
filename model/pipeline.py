# model/pipeline.py

"""
백엔드 팀이 사용할 최종 딥페이크 탐지 파이프라인.

사용법:
    from model.pipeline import DeepfakeDetectionPipeline

    pipeline = DeepfakeDetectionPipeline()        # 서버 시작 시 1회만
    result = pipeline.run(pil_image)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL_NAME
from .inference import load_model, predict_from_pil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class DeepfakeDetectionPipeline:
    """
    이미지 1장을 받아서 딥페이크 여부를 반환하는 파이프라인.

    반환값 예시:
    {
        "success": True,
        "label": "FAKE",
        "fake_probability": 0.91,
        "real_probability": 0.09,
        "confidence": "high",
        "confidence_score": 0.82,
    }

    에러 시:
    {
        "success": False,
        "error": "no_face",
        "message": "얼굴을 찾을 수 없습니다",
    }
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_face_crop: bool = True,
        device: Optional[str] = None,
    ):
        print(f"[INFO] 딥페이크 탐지 모델 로드 중: {model_name}")
        self.model = load_model(model_name=model_name, device=device)
        self.device = device
        self.use_face_crop = use_face_crop

        if use_face_crop:
            self._init_face_detector()

        print("[INFO] 파이프라인 준비 완료")

    def _init_face_detector(self):
        try:
            from .face_detectors import make_detector
            self.face_detector = make_detector("mtcnn")
            print("[INFO] 얼굴 탐지기 로드 완료")
        except Exception as e:
            print(f"[WARN] 얼굴 탐지기 로드 실패, 전체 이미지로 진행합니다: {e}")
            self.use_face_crop = False

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
            result = predict_from_pil(model=self.model, img=img, device=self.device)
        except Exception as e:
            return {
                "success": False,
                "error": "model_error",
                "message": f"모델 오류: {e}",
            }

        fake_prob = result["fake_probability"]
        real_prob = result["real_probability"]
        confidence_score = abs(fake_prob - 0.5) * 2  # 0~1

        return {
            "success": True,
            "label": result["label"],
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": self._confidence_level(confidence_score),
            "confidence_score": round(confidence_score, 4),
        }

    def _crop_face(self, img: Image.Image) -> Optional[Image.Image]:
        """얼굴 영역만 crop해서 반환. 실패 시 None."""
        try:
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
        elif score < 0.5:
            return "medium"
        return "high"
