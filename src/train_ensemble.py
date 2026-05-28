# train_ensemble.py

import argparse
from pathlib import Path
import sys
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Resolve Runtime Module Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model import create_model
from src.dataset import DeepfakeDataset


# ==========================================
# 1. Custom Optimization Loss Definition
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """Focal Loss for robust learning against compressed hard web samples."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==========================================
# 2. Environment Path Configurations
# ==========================================
LOCAL_DATA_DIR = Path("/content/dataset")
DRIVE_DATA_DIR = PROJECT_ROOT / "dataset"

# Automated path setup logic optimizing loading speeds
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else DRIVE_DATA_DIR
print(f"[System] Runtime environment mapping data repository to: {DATA_DIR}")

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. Training & Validation Core Loops
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="  🚀 Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  🔍 Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ==========================================
# 4. Main Engine Pipeline
# ==========================================
def run_training(args):
    session_info = f"Domain: {args.domain.upper()} | Backbone: {args.arch}"
    print(f"\n{'='*60}\n[RUNNING SESSION] {session_info}\n{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Activating processing compute core: {device}")
    
    # 4.1 Dataset Provisioning
    train_csv = DATA_DIR / f"{args.domain}_train.csv"
    val_csv = DATA_DIR / f"{args.domain}_val.csv"

    if not train_csv.exists():
        print(f"[Runtime Error] Staging file {train_csv.name} not found.")
        return

    train_dataset = DeepfakeDataset(train_csv, train=True)
    val_dataset = DeepfakeDataset(val_csv, train=False)

    # 4.2 Data Loader Provisioning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"[System] Pipelines loaded (Batch Size: {args.batch_size} | Threads: {args.num_workers})")

    # 4.3 Model Initialization
    model = create_model(arch=args.arch, num_classes=2, dropout_rate=args.dropout).to(device)
    
    # 4.4 Load Checkpoint Weights (Resume Session)
    if args.resume_from:
        resume_path = WEIGHTS_DIR / args.resume_from
        if resume_path.exists():
            print(f"[System] Resuming checkpoint artifact weights from: {resume_path.name}")
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"[Runtime Error] Target checkpoint missing: {resume_path}")
            sys.exit(1)

    # 4.5 Optimizer, Criterion, & Scheduler Settings
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_name = f"{args.domain}_{args.arch}_focal.pth"
    output_path = WEIGHTS_DIR / output_name
    best_acc = 0.0

    # 4.6 Execution Loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} || "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f} | "
              f"Duration: {elapsed:.1f}s")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), output_path)
            print(f"  --> 💾 Checkpoint saved: {output_name} (New Best Accuracy Established)")

    print(f"\n[Terminated] Session execution complete. Peak Valuation Accuracy: {best_acc:.4f}")


# ==========================================
# 5. Argument Interface Parser
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens Deepfake Detection Strategy Model Training Engine")
    
    # ✂️ Cleaned legacy architectures to align with core deployment choices
    parser.add_argument("--arch", type=str, default="efficientnet_b0", 
                        choices=["efficientnet_b0", "mobilenet_v3"],
                        help="Target backbone network architecture selection")
    parser.add_argument("--domain", type=str, default="dfdc", choices=["celebdf", "dfdc"],
                        help="Dataset source domain framework mapping context")
    parser.add_argument("--resume-from", type=str, default="", help="Filename target to resume checkpoints from")
    parser.add_argument("--num-workers", type=int, default=2, help="Compute context thread worker allocation count")
    parser.add_argument("--epochs", type=int, default=10, help="Total system operational training cycle limit")
    parser.add_argument("--batch-size", type=int, default=32, help="Payload data matrix batch processing capacity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial gradient descend learning coefficient metric")
    parser.add_argument("--dropout", type=float, default=0.5, help="Regularization channel matrix layer dropout probability")

    args = parser.parse_args()
    run_training(args)