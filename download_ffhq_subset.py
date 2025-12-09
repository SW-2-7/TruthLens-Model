# download_ffhq_subset.py
import os
import random
import urllib.request
from pathlib import Path

# 저장 위치
OUT_DIR = Path("data/external/ffhq_2000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FFHQ 해상도 선택 (일반적으로 1024 사용)
BASE_URL = "https://github.com/NVlabs/ffhq-dataset/raw/master/downloads/FFHQ_1024/images/{:05d}.png"

# 총 70,000장 중 원하는 개수
COUNT = 2000

def download_image(img_id):
    url = BASE_URL.format(img_id)
    out_path = OUT_DIR / f"{img_id:05d}.png"
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"[OK] {out_path}")
    except Exception as e:
        print(f"[FAIL] {url} ({e})")

def main():
    # 0~69999 중에서 2000장 랜덤 선택
    ids = random.sample(range(70000), COUNT)

    print(f"[INFO] Downloading {COUNT} FFHQ images to: {OUT_DIR}\n")

    for img_id in ids:
        download_image(img_id)

    print("\n[INFO] Done!")

if __name__ == "__main__":
    main()
