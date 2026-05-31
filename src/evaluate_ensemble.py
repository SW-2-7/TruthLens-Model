# evaluate_ensemble.py

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

# Resolve Runtime Module Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model import create_model
from src.dataset import DeepfakeDataset

# ==========================================
# Environment Path Configurations
# ==========================================
LOCAL_DATA_DIR = Path("/content/dataset")
DRIVE_DATA_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else DRIVE_DATA_DIR
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def load_trained_model(arch: str, domain: str, device) -> nn.Module:
    """Loads specific checkpoint artifacts matched with custom Focal Loss naming naming conventions."""
    model_path = WEIGHTS_DIR / f"{domain}_{arch}_focal.pth"
    model = create_model(arch=arch, num_classes=2)

    if not model_path.exists():
        print(f"[Warning] Weight missing for checkpoint: {model_path.name}")
        return None

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Runtime Error] Initialization failed for {arch}: {e}")
        return None


def evaluate_models(args):
    print(f"\n{'='*60}")
    print(f"🎯 Ensemble Performance Evaluation (Test Suite: {args.target_domain.upper()})")
    print(f"📊 Target Data Source: {DATA_DIR}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Pipeline Dataset Loading
    test_csv = DATA_DIR / f"{args.target_domain}_test.csv"
    if not test_csv.exists():
        print(f"[Runtime Error] Missing evaluation indexing source: {test_csv.name}")
        return

    test_dataset = DeepfakeDataset(test_csv, train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    print(f"📊 [System] Total Evaluation Payload Volume: {len(test_dataset)} images")

    # 2. Extract Core Backend Model Implementations
    model_eff = load_trained_model("efficientnet_b0", args.trained_domain, device)
    model_mob = load_trained_model("mobilenet_v3", args.trained_domain, device)

    if model_eff is None or model_mob is None:
        print("[Error] Failed to initialize model backbones. Check checkpoint weight filenames.")
        return

    all_labels = []
    probs_eff = []
    probs_mob = []
    probs_ensemble = []

    # 🚀 Core Weighted Ensemble Configuration Ratio (8:2 Split Strategy)
    alpha = 0.8  # Weight multiplier for EfficientNet-B0 (Primary)
    beta = 0.2   # Weight multiplier for MobileNet-V3 (Support)
    
    print(f"🔗 [System] Soft Voting Configuration Initialized -> (Eff-B0: {alpha} | Mob-V3: {beta})")

    # 3. Execution Phase
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            all_labels.extend(labels.numpy())

            # Primary Ensemble Feed: EfficientNet-B0
            out_eff = model_eff(images)
            prob_e = torch.softmax(out_eff, dim=1)[:, 1].cpu().numpy()
            probs_eff.extend(prob_e)

            # Secondary Ensemble Feed: MobileNet-V3
            out_mob = model_mob(images)
            prob_m = torch.softmax(out_mob, dim=1)[:, 1].cpu().numpy()
            probs_mob.extend(prob_m)

            # Execution Fusion (Weighted Averaging)
            prob_ens = (alpha * prob_e) + (beta * prob_m)
            probs_ensemble.extend(prob_ens)

    # 4. Metrics Reporting Function Block
    def print_metrics(name: str, probs_array: list, true_labels: list):
        probs_array = np.array(probs_array)
        true_labels = np.array(true_labels)
        preds = (probs_array >= 0.5).astype(int)
        
        acc = np.mean(preds == true_labels)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        auc = roc_auc_score(true_labels, probs_array) if len(np.unique(true_labels)) > 1 else 0.0

        print(f"[{name}]")
        print(f"  - Accuracy  : {acc*100:.2f}%")
        print(f"  - ROC-AUC   : {auc:.4f}")
        print(f"  - Precision : {prec*100:.2f}%")
        print(f"  - Recall    : {rec*100:.2f}%")
        print(f"  - F1-Score  : {f1*100:.2f}%\n")

    print(f"\n{'='*60}\n📊 Final Ensemble Evaluation Dashboard Summary\n{'='*60}")
    print_metrics("1. EfficientNet-B0 Backbone (Standalone)", probs_eff, all_labels)
    print_metrics("2. MobileNet-V3 Backbone (Standalone)", probs_mob, all_labels)
    print_metrics("🌟 3. Integrated Production Hybrid Ensemble (Final Outcome)", probs_ensemble, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens Deepfake Detection System Model Evaluation Engine")
    parser.add_argument("--trained-domain", type=str, default="dfdc", choices=["celebdf", "dfdc"],
                        help="Trained checkpoint weight origin mapping domain context")
    parser.add_argument("--target-domain", type=str, default="dfdc", choices=["celebdf", "dfdc"],
                        help="Target dataset framework domain used to audit validation metrics")
    parser.add_argument("--batch-size", type=int, default=64, help="Payload matrix data pipeline chunk size")
    args = parser.parse_args()

    evaluate_models(args)


