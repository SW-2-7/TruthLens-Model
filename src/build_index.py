# build_index.py

from pathlib import Path
import os
import random
import cv2
import pandas as pd

# ============================================
# 1. Global Configurations & Environment
# ============================================
SEED = 42
random.seed(SEED)

# Define dataset hierarchy paths
BASE_DATA_DIR = Path.cwd() / "dataset"
DATA_DIRS = {
    "celebdf": {
        "real": BASE_DATA_DIR / "Celeb_real_face_only",
        "fake": BASE_DATA_DIR / "Celeb_fake_face_only",
    },
    "dfdc": {
        "real": BASE_DATA_DIR / "DFDC_REAL_Face_only_data",
        "fake": BASE_DATA_DIR / "DFDC_FAKE_Face_only_data",
    },
}

# Train : Validation : Test Split Ratio (8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# ============================================
# 2. Video Processing Pipeline
# ============================================
def extract_frames_from_video(video_path: Path, num_frames: int = 5) -> None:
    """Extracts a fixed number of evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    step = max(1, total_frames // num_frames)
    count = 0
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:
            save_path = video_path.parent / f"{video_path.stem}_frame{count}.jpg"
            # Optimization: Skip extraction if the frame target already exists
            if not save_path.exists():
                cv2.imwrite(str(save_path), frame)
            count += 1
            if count >= num_frames:
                break
                
    cap.release()


# ============================================
# 3. Data Indexing & Split Utilities
# ============================================
def get_image_paths(folder_path: Path) -> list:
    """Gathers all image paths within a folder matching specific extension formats."""
    if not folder_path.exists():
        print(f"[Warning] Directory target not found: {folder_path.name}")
        return []

    valid_extensions = (".png", ".jpg", ".jpeg")
    paths = [
        str(p) for p in folder_path.rglob("*.*") 
        if p.suffix.lower() in valid_extensions
    ]
    print(f"  -> Found {len(paths)} images inside '{folder_path.name}'")
    return paths


def split_by_ratio(data_list: list) -> tuple:
    """Splits an input dataset list into Train, Validation, and Test subsets."""
    total = len(data_list)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    return data_list[:train_end], data_list[train_end:val_end], data_list[val_end:]


# ============================================
# 4. Main Executive Engine
# ============================================
def main():
    # --- STEP 1: Video-to-Frame Data Generation ---
    print("🚀 [Step 1] Initiating Video-to-JPG Frame Extraction...")
    for domain, folders in DATA_DIRS.items():
        for label_type in ["real", "fake"]:
            folder = folders[label_type]
            if not folder.exists(): 
                continue
            
            mp4_files = list(folder.rglob("*.mp4"))
            if mp4_files:
                print(f"📁 Processing [{folder.name}] -> Found {len(mp4_files)} videos.")
                for idx, mp4_path in enumerate(mp4_files):
                    extract_frames_from_video(mp4_path, num_frames=5)
                    if (idx + 1) % 100 == 0:
                        print(f"  - Progress: {idx + 1}/{len(mp4_files)} video streams complete.")

    # --- STEP 2: Dataset Auditing & CSV Generation ---
    print("\n🚀 [Step 2] Splitting Datasets & Generating Index CSV Files...")
    if not BASE_DATA_DIR.exists():
        print(f"[Error] Target base path {BASE_DATA_DIR} does not exist.")
        return

    for domain_name, paths in DATA_DIRS.items():
        print(f"\nIndexing Domain Suite: [{domain_name.upper()}]")
        real_images = get_image_paths(paths["real"])
        fake_images = get_image_paths(paths["fake"])

        if not real_images and not fake_images:
            print(f"  - [Notice] No image payloads found for domain: {domain_name}")
            continue

        # Shuffle lists to avoid sequential cluster dependency bias
        random.shuffle(real_images)
        random.shuffle(fake_images)

        r_tr, r_val, r_te = split_by_ratio(real_images)
        f_tr, f_val, f_te = split_by_ratio(fake_images)

        splits = {
            "train": (r_tr, f_tr), 
            "val": (r_val, f_val), 
            "test": (r_te, f_te)
        }

        for split_name, (r_list, f_list) in splits.items():
            data = []
            # Mapping Target -> REAL: 0, FAKE: 1
            for img in r_list: 
                data.append({"image_path": img, "label": 0, "domain": domain_name})
            for img in f_list: 
                data.append({"image_path": img, "label": 1, "domain": domain_name})

            df = pd.DataFrame(data)
            if df.empty:
                print(f"  - [Warning] Split bucket '{split_name}' has insufficient data volume.")
                continue

            # Full shuffle before flattening to disk
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            csv_filename = BASE_DATA_DIR / f"{domain_name}_{split_name}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"  - Generated Index Metafile: {csv_filename.name} (Total: {len(df)} images)")

    print("\n✅ All data pipelines executed successfully! Environment ready for training.")


if __name__ == "__main__":
    main()