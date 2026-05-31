# model/face_detectors.py

"""
Face detector adapters used by the preprocessing pipeline.

Default behavior is intentionally safe for the current project:
- `auto` tries InsightFace RetinaFace first, then MTCNN, then OpenCV Haar.
- If optional packages are not installed, the code falls back automatically.
- The output format is unified so extract_faces.py does not depend on a specific detector.

Optional installs:
    pip install insightface onnxruntime
    pip install facenet-pytorch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import math
import time

import cv2
import numpy as np


@dataclass
class Detection:
    """Unified face detection result."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    detector: str
    landmarks: Optional[np.ndarray] = None  # shape: (5, 2), if available

    @property
    def area(self) -> int:
        return int(self.bbox[2] * self.bbox[3])


class BaseFaceDetector:
    name = "base"

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        raise NotImplementedError


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar Cascade fallback detector."""

    name = "haar"

    def __init__(self, min_size: int = 40):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.min_size = min_size
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_size, self.min_size),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        return Detection(bbox=(int(x), int(y), int(w), int(h)), score=1.0, detector=self.name)


class InsightFaceDetector(BaseFaceDetector):
    """InsightFace FaceAnalysis detector. This uses RetinaFace-style detection internally."""

    name = "insightface"

    def __init__(self, det_size: int = 640, ctx_id: int = -1):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is not installed. Install with: pip install insightface onnxruntime"
            ) from e

        # CPU provider is the most portable default. Users can still install onnxruntime-gpu,
        # but this pipeline should not fail on a CPU-only machine.
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        faces = self.app.get(img_bgr)
        if not faces:
            return None

        def rank(face) -> float:
            x1, y1, x2, y2 = face.bbox.astype(float)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return area * float(getattr(face, "det_score", 1.0))

        face = max(faces, key=rank)
        x1, y1, x2, y2 = face.bbox.astype(float)
        x, y = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        landmarks = getattr(face, "kps", None)
        if landmarks is not None:
            landmarks = np.asarray(landmarks, dtype=np.float32)
        return Detection(
            bbox=(x, y, w, h),
            score=float(getattr(face, "det_score", 1.0)),
            detector=self.name,
            landmarks=landmarks,
        )


class MTCNNFaceDetector(BaseFaceDetector):
    """facenet-pytorch MTCNN fallback detector."""

    name = "mtcnn"

    def __init__(self, min_face_size: int = 40, device: str = "cpu"):
        try:
            from facenet_pytorch import MTCNN
        except ImportError as e:
            raise ImportError("facenet-pytorch is not installed. Install with: pip install facenet-pytorch") from e

        self.mtcnn = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=min_face_size,
            post_process=False,
            keep_all=True,
            device=device,
        )

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None

        probs = np.asarray(probs, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32)
        landmarks = np.asarray(landmarks, dtype=np.float32) if landmarks is not None else None

        def rank(i: int) -> float:
            x1, y1, x2, y2 = boxes[i]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            score = float(probs[i]) if not math.isnan(float(probs[i])) else 0.0
            return area * score

        idx = max(range(len(boxes)), key=rank)
        x1, y1, x2, y2 = boxes[idx]
        x, y = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        lm = landmarks[idx] if landmarks is not None else None
        score = float(probs[idx]) if not math.isnan(float(probs[idx])) else 0.0
        return Detection(bbox=(x, y, w, h), score=score, detector=self.name, landmarks=lm)


class AutoFaceDetector(BaseFaceDetector):
    """Detector chain. It tries each available detector until one returns a face."""

    name = "auto"

    def __init__(self, order: Iterable[str] = ("insightface", "mtcnn", "haar"), **kwargs):
        self.detectors: list[BaseFaceDetector] = []
        errors: list[str] = []

        for detector_name in order:
            try:
                detector = make_detector(detector_name, **kwargs)
                self.detectors.append(detector)
            except Exception as e:
                errors.append(f"{detector_name}: {e}")

        if not self.detectors:
            raise RuntimeError("No detector could be initialized. Errors: " + " | ".join(errors))

        if errors:
            print("[WARN] Some detector backends were skipped:")
            for msg in errors:
                print(f"       - {msg}")
        print("[INFO] Active detector chain:", " -> ".join(d.name for d in self.detectors))

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        for detector in self.detectors:
            result = detector.detect_main_face(img_bgr)
            if result is not None:
                return result
        return None


def make_detector(name: str = "auto", **kwargs) -> BaseFaceDetector:
    """Factory for face detector backends."""
    normalized = name.lower().strip()

    if normalized == "auto":
        return AutoFaceDetector(**kwargs)
    if normalized in {"haar", "opencv", "cascade"}:
        return HaarFaceDetector(min_size=int(kwargs.get("min_size", 40)))
    if normalized in {"insightface", "retinaface", "retina"}:
        return InsightFaceDetector(det_size=int(kwargs.get("det_size", 640)), ctx_id=int(kwargs.get("ctx_id", -1)))
    if normalized == "mtcnn":
        return MTCNNFaceDetector(min_face_size=int(kwargs.get("min_size", 40)), device=str(kwargs.get("device", "cpu")))

    raise ValueError(f"Unsupported detector: {name}")


def clip_bbox(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image_width, int(x + w))
    y2 = min(image_height, int(y + h))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def expand_to_square_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: float = 0.25,
) -> tuple[int, int, int, int]:
    """Expand a face bbox to a square crop with margin."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + float(margin))

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


# ArcFace 5-point reference landmarks for a 112x112 aligned face.
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face_by_landmarks(
    img_bgr: np.ndarray,
    landmarks: Optional[np.ndarray],
    output_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Align a face using 5 landmarks if available.

    This is useful for deepfake classification because the model receives
    more consistent eye/nose/mouth positions across frames.
    """
    if landmarks is None:
        return None

    src = np.asarray(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        return None

    dst = _ARCFACE_REF_112 * (float(output_size) / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        return None

    aligned = cv2.warpAffine(img_bgr, transform, (output_size, output_size), flags=cv2.INTER_LINEAR)
    if aligned is None or aligned.size == 0:
        return None
    return aligned


def crop_face(
    img_bgr: np.ndarray,
    detection: Detection,
    output_size: int = 224,
    margin: float = 0.25,
    align: bool = True,
) -> Optional[np.ndarray]:
    """Return an aligned face if possible; otherwise return square bbox crop."""
    if align and detection.landmarks is not None:
        aligned = align_face_by_landmarks(img_bgr, detection.landmarks, output_size=output_size)
        if aligned is not None:
            return aligned

    image_height, image_width = img_bgr.shape[:2]
    x, y, w, h = expand_to_square_bbox(detection.bbox, image_width, image_height, margin=margin)
    x, y, w, h = clip_bbox(x, y, w, h, image_width, image_height)
    if w <= 0 or h <= 0:
        return None

    face = img_bgr[y : y + h, x : x + w]
    if face.size == 0:
        return None
    return cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_AREA)


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
