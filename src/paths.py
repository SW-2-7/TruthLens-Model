# src/paths.py

from pathlib import Path

# Automatically resolves the absolute project root directory relative to this script
# PROJECT_ROOT resolves to: .../TruthLens-Model/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1. Dataset Directory (Stores raw face image folders and indexing CSV metafiles)
DATA_DIR = PROJECT_ROOT / "dataset"

# 2. Weights Directory (Stores trained backup .pth checkpoint artifacts)
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Optimization: Safety execution hook to prevent directory missing exceptions
if not WEIGHTS_DIR.exists():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[System] Initialized missing directory repository at: {WEIGHTS_DIR.name}/")