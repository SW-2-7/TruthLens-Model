# model/model.py

from torchvision import models
import torch.nn as nn


def create_model(arch: str = 'efficientnet_b0', num_classes: int = 2, dropout_rate: float = 0.5) -> nn.Module:
    """Factory function to build core deepfake detection backbone architectures."""
    
    # 1. Lightweight Backbone: MobileNet-V3 Large
    if arch == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    # 2. Primary Production Backbone: EfficientNet-B0 (DFDC/Celeb-DF Optimizer)
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    else:
        raise ValueError(f"Unsupported model architecture token: {arch}")
        
    return model