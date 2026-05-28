# dataset.py

from pathlib import Path
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    def __init__(self, csv_file: str, train: bool = True):
        """Custom Dataset for loading Deepfake facial image meta-files."""
        self.data = pd.read_csv(csv_file)
        self.train = train
        
        # Standard ImageNet Normalization Parameters
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        if self.train:
            # Robust Augmentation Suite to handle compressed DFDC environments
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                # 1. Color Jittering (Enhances variance for dynamic lighting)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # 2. Random Rotation (Compensates for head tilt deviations)
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
                # 3. Random Erasing (Forces attention to micro-textures rather than whole ROI - AUC Optimizer)
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        # 1. Parse raw path from csv metadata
        raw_path = str(self.data.iloc[idx]['image_path'])
        
        # 2. Standardize OS path separators to POSIX standard (Linux/Mac/Colab compatible)
        clean_path = raw_path.replace('\\', '/')
        
        # 3. 🚀 Environment Agnostic Path Resolution (Handles Colab Local, Drive, or Native Environments)
        if 'dataset/' in clean_path:
            relative_target = clean_path.split('dataset/')[-1]
            
            # Setup dynamic fallbacks based on runtime environment existence
            local_candidate = Path(f"/content/dataset/{relative_target}")
            drive_candidate = Path(f"/content/drive/MyDrive/TruthLens-Model/dataset/{relative_target}")
            native_candidate = Path("dataset") / relative_target
            
            if local_candidate.exists():
                img_path = local_candidate
            elif drive_candidate.exists():
                img_path = drive_candidate
            else:
                img_path = native_candidate  # Defaults to relative project path for local/server users
        else:
            img_path = Path(clean_path)

        label = int(self.data.iloc[idx]['label'])
    
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Graceful Fallback: Generate a black dummy image if stream is corrupted or missing
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        return self.transform(image), label