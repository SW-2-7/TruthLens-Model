# model/preprocess.py

from typing import Callable
import torch
from torchvision import transforms
from PIL import Image

_transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """
    PIL 이미지를 받아서 모델 입력 tensor(1, 3, 224, 224)로 변환
    """
    # 혹시 RGBA 등 들어오면 RGB로 통일
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = _transform(img)  # (3, 224, 224)
    x = x.unsqueeze(0)   # (1, 3, 224, 224)
    return x
