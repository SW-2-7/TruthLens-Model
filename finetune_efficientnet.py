# finetune_efficientnet.py
#
# DeepFake-Eval-2024 train 셋으로 EfficientNet-B0을 파인튜닝합니다.
# 학습 후 자동으로 test 셋 평가까지 진행합니다.
#
# 사용법:
#   python finetune_efficientnet.py `
#       --weights     add/dfdc_efficientnet_b0_focal.pth `
#       --dataset-dir data/deepfake_eval_2024 `
#       --output-name eff_b0_finetuned.pth
#
# 주요 옵션:
#   --epochs        학습 에폭 수 (기본: 10)
#   --batch-size    배치 크기 (기본: 32)
#   --lr            학습률 (기본: 1e-4)
#   --max-train     train 셋 최대 영상 수, 0=전체 (기본: 0)
#   --frames-per-video  영상당 프레임 수 (기본: 8)

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────
# 1. 메타데이터 로드
# ──────────────────────────────────────────────
def load_metadata(metadata_csv: Path, video_dir: Path,
                  split: str, max_videos: int = 0) -> list[dict]:
    records = []
    skipped = 0
    with metadata_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split != "all" and row.get("Finetuning Set","").strip().lower() != split:
                continue
            gt = row.get("Video Ground Truth","").strip().lower()
            if gt == "fake":   label = 1
            elif gt == "real": label = 0
            else:
                skipped += 1
                continue
            vp = video_dir / row.get("Filename","").strip()
            if not vp.exists():
                skipped += 1
                continue
            records.append({"video_path": vp, "label": label})

    if skipped:
        print(f"[INFO] 건너뛴 영상: {skipped}개")
    n_real = sum(1 for r in records if r["label"]==0)
    n_fake = sum(1 for r in records if r["label"]==1)
    print(f"[INFO] {split} 셋: 총 {len(records)}개 (Real={n_real}, Fake={n_fake})")

    if max_videos > 0 and len(records) > max_videos:
        reals = random.sample([r for r in records if r["label"]==0],
                               min(max_videos//2, n_real))
        fakes = random.sample([r for r in records if r["label"]==1],
                               min(max_videos//2, n_fake))
        records = reals + fakes
        random.shuffle(records)
        print(f"[INFO] 샘플링 후: {len(records)}개")

    return records


# ──────────────────────────────────────────────
# 2. 프레임 추출
# ──────────────────────────────────────────────
def sample_indices(total: int, k: int) -> list[int]:
    if total <= 0: return []
    if total <= k: return list(range(total))
    return [int(i*(total-1)/(k-1)) for i in range(k)]

def extract_frames(video_path: Path, out_dir: Path, k: int) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    for idx in sample_indices(total, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: continue
        out_path = out_dir / f"{video_path.stem}_f{idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    cap.release()
    return saved


# ──────────────────────────────────────────────
# 3. MTCNN 얼굴 크롭
# ──────────────────────────────────────────────
def make_mtcnn(device: str):
    try:
        from facenet_pytorch import MTCNN
        return MTCNN(image_size=224, margin=0, min_face_size=40,
                     post_process=False, keep_all=False, device=device)
    except ImportError:
        print("[WARN] facenet-pytorch 없음 → 원본 프레임 사용")
        return None

def crop_face(mtcnn, img: Image.Image) -> Image.Image:
    if mtcnn is None: return img
    try:
        face = mtcnn(img)
        if face is None: return img
        return Image.fromarray(face.permute(1,2,0).numpy().astype("uint8"))
    except Exception:
        return img


# ──────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, frame_records: list[dict], train: bool = True):
        self.records = frame_records
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["frame_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224))
        return self.transform(img), torch.tensor(rec["label"], dtype=torch.long)


# ──────────────────────────────────────────────
# 5. 모델 로드
# ──────────────────────────────────────────────
def _extract_sd(ckpt):
    if not isinstance(ckpt, dict): return ckpt
    for k in ("model_state_dict","state_dict","model"):
        if k in ckpt and isinstance(ckpt[k], dict): return ckpt[k]
    return ckpt

def _strip_dp(sd: dict) -> dict:
    if sd and all(k.startswith("module.") for k in sd):
        return {k[7:]: v for k,v in sd.items()}
    return sd

def load_efficientnet(weights: Path, device: str) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, 2)
    )
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    m.load_state_dict(_strip_dp(_extract_sd(ckpt)))
    return m.to(device)


# ──────────────────────────────────────────────
# 6. 학습 / 평가 루프
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss/max(1,total), correct/max(1,total)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        probs = torch.softmax(out, dim=1)[:,1]
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    auc = roc_auc_score(all_labels, all_probs) \
          if len(set(all_labels)) > 1 else 0.0
    return total_loss/max(1,total), correct/max(1,total), float(auc)


# ──────────────────────────────────────────────
# 7. 메인
# ──────────────────────────────────────────────
def run(
    weights: Path,
    dataset_dir: Path,
    output_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    frames_per_video: int,
    max_train: int,
    num_workers: int,
    amp: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")
    print(f"[INFO] 원본 가중치: {weights.name}")

    video_dir    = dataset_dir / "video-data"
    metadata_csv = dataset_dir / "video-metadata-publish-with-links.csv"
    frames_dir   = dataset_dir / "_frames_cache"
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_dir = PROJECT_ROOT / "model" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    meta_path   = output_path.with_suffix(".finetune.json")

    # ── 메타데이터 로드 ──
    print("\n[1/4] 데이터 준비...")
    train_records = load_metadata(metadata_csv, video_dir, "train", max_train)
    test_records  = load_metadata(metadata_csv, video_dir, "test",  0)

    if not train_records:
        print("[ERROR] train 데이터가 없습니다.")
        return

    # ── 프레임 추출 ──
    print("  프레임 추출 중 (캐시 있으면 스킵)...")
    mtcnn = make_mtcnn(device)

    def build_frame_records(video_records, tag):
        out = []
        for rec in tqdm(video_records, desc=f"  {tag}"):
            frames = extract_frames(rec["video_path"], frames_dir, frames_per_video)
            for fp in frames:
                # MTCNN 크롭 후 덮어쓰기
                try:
                    img = Image.open(fp).convert("RGB")
                    face = crop_face(mtcnn, img)
                    face.save(fp)
                except Exception:
                    pass
                out.append({"frame_path": str(fp), "label": rec["label"]})
        return out

    train_frames = build_frame_records(train_records, "train")
    test_frames  = build_frame_records(test_records,  "test")
    print(f"  train 프레임: {len(train_frames)}개 | test 프레임: {len(test_frames)}개")

    # ── DataLoader ──
    train_ds = FrameDataset(train_frames, train=True)
    test_ds  = FrameDataset(test_frames,  train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(device=="cuda"))

    # ── 모델 로드 ──
    print("\n[2/4] 모델 로드...")
    model = load_efficientnet(weights, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    scaler = torch.cuda.amp.GradScaler() \
             if amp and device == "cuda" else None

    # ── 학습 ──
    print(f"\n[3/4] 파인튜닝 시작 (epochs={epochs}, lr={lr}, bs={batch_size})...")
    best_auc  = -1.0
    best_acc  = 0.0
    history   = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_auc = eval_one_epoch(
            model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(tr_loss,4),
                   train_acc=round(tr_acc,4), val_loss=round(val_loss,4),
                   val_acc=round(val_acc,4), val_auc=round(val_auc,4),
                   lr=round(scheduler.get_last_lr()[0], 6),
                   elapsed_sec=round(elapsed,1))
        history.append(row)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
              f"{elapsed:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            torch.save({
                "arch"            : "efficientnet_b0",
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "best_val_auc"    : best_auc,
                "best_val_acc"    : best_acc,
            }, output_path)
            print(f"  --> Best 저장: {output_path.name} "
                  f"(AUC={best_auc:.4f}, Acc={best_acc:.4f})")

    # ── 최종 결과 ──
    print(f"\n[4/4] 파인튜닝 완료")
    print(f"  Best val AUC : {best_auc:.4f}")
    print(f"  Best val Acc : {best_acc:.4f}")
    print(f"  저장 경로    : {output_path}")

    meta = {
        "arch"         : "efficientnet_b0",
        "base_weights" : str(weights),
        "output_path"  : str(output_path),
        "dataset"      : str(dataset_dir),
        "best_val_auc" : best_auc,
        "best_val_acc" : best_acc,
        "history"      : history,
        "args": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "label_smoothing": label_smoothing,
            "frames_per_video": frames_per_video, "max_train": max_train,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터   : {meta_path}")

    print("\n파인튜닝된 모델로 eval_deepfake_eval_2024.py를 다시 실행해 성능을 확인하세요.")
    print(f"  --eff-weights model/weights/{output_name}")


# ──────────────────────────────────────────────
# 8. CLI
# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="EfficientNet-B0 DeepFake-Eval-2024 파인튜닝"
    )
    p.add_argument("--weights",      required=True,
                   help="기존 EfficientNet-B0 가중치 경로")
    p.add_argument("--dataset-dir",  default="data/deepfake_eval_2024")
    p.add_argument("--output-name",  default="eff_b0_finetuned.pth",
                   help="저장할 가중치 파일명 (model/weights/ 아래)")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--max-train",    type=int,   default=0,
                   help="train 최대 영상 수 (0=전체)")
    p.add_argument("--num-workers",  type=int,   default=0)
    p.add_argument("--amp",          action="store_true",
                   help="CUDA mixed precision 사용")
    args = p.parse_args()

    def res(s: str) -> Path:
        path = Path(s)
        return path if path.is_absolute() else PROJECT_ROOT / path

    run(
        weights         = res(args.weights),
        dataset_dir     = res(args.dataset_dir),
        output_name     = args.output_name,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        frames_per_video= args.frames_per_video,
        max_train       = args.max_train,
        num_workers     = args.num_workers,
        amp             = args.amp,
    )

if __name__ == "__main__":
    main()