# model/face_crop.py

from typing import Optional, Tuple

from PIL import Image
import torch
from facenet_pytorch import MTCNN

# 전역 MTCNN 인스턴스 (한 번만 초기화)
_mtcnn: Optional[MTCNN] = None


def get_mtcnn(device: str = "cpu") -> MTCNN:
    """
    lazy init 방식으로 MTCNN 생성
    """
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=False, device=device)  # 가장 큰 얼굴 1개만
    return _mtcnn


def detect_and_crop_face(
    img: Image.Image,
    device: str = "cpu",
    margin: float = 0.2,
    min_face_size: int = 40,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    """
    PIL 이미지를 받아서:
    - MTCNN으로 가장 큰 얼굴 탐지
    - margin 비율만큼 bbox 확장
    - 얼굴 영역만 crop 해서 PIL.Image로 반환

    return:
        (cropped_face_img, bbox) 또는 (None, None)
    """
    mtcnn = get_mtcnn(device=device)

    # MTCNN은 tensor/ndarray도 받지만, PIL도 바로 지원
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return None, None

    # 가장 큰 얼굴 선택
    boxes = boxes.tolist()
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    x1, y1, x2, y2 = boxes[0]

    # 너무 작은 얼굴이면 무시 (예: min_face_size 이하)
    if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
        return None, None

    # margin 만큼 확장
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * (1 + margin)
    new_h = h * (1 + margin)

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(img.width, cx + new_w / 2))
    ny2 = int(min(img.height, cy + new_h / 2))

    cropped = img.crop((nx1, ny1, nx2, ny2))

    return cropped, (nx1, ny1, nx2, ny2)
