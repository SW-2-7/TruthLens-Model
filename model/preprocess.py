# preprocess.py

from typing import Callable
from PIL import Image
from torchvision import transforms
import torch

# 🚀 ImageNet Pre-trained Normalization Parameters
IMAGE_NET_TRANSFORM: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Converts a PIL Image into a standardized model input tensor (1, 3, 224, 224).
    
    Args:
        img (Image.Image): Input PIL Image.
        
    Returns:
        torch.Tensor: Preprocessed and normalized tensor with a batch dimension.
    """
    # Force convert to RGB mode to eliminate alpha channels (RGBA) or grayscale issues
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = IMAGE_NET_TRANSFORM(img)  # Shape: (3, 224, 224)
    x = x.unsqueeze(0)            # Shape: (1, 3, 224, 224)
    return x