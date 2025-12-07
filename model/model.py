# model/model.py

from typing import Literal

import torch.nn as nn
from torchvision import models

ArchName = Literal["resnet18", "resnet50"]


def create_model(arch: ArchName = "resnet18", num_classes: int = 2) -> nn.Module:
    """
    arch: "resnet18" 또는 "resnet50"
    num_classes: 출력 클래스 수 (이진 분류면 2)
    """
    if arch == "resnet18":
        # 학습된 weight를 따로 로드할 것이기 때문에 weights=None 권장
        backbone = models.resnet18(weights=None)
    elif arch == "resnet50":
        backbone = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    return backbone
