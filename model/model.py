# model/model.py

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, resnet50


ArchName = Literal["resnet18", "resnet50"]


def create_model(arch: ArchName = "resnet18", num_classes: int = 2) -> nn.Module:
    """
    arch: "resnet18" 또는 "resnet50"
    num_classes: 출력 클래스 수 (이진 분류면 2)
    """
    if arch == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    return backbone
