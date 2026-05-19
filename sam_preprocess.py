"""
SAM Background Removal Preprocessing (uses Ultralytics MobileSAM — auto-downloads)
Segments spice from background → saves to Spice_Spectrum_SAM/

Setup:
    pip install ultralytics

Usage:
    python sam_preprocess.py                   # processes all classes
    python sam_preprocess.py --cls turmeric    # single class (for testing)
"""
import sys
import os
sys.path.insert(0, "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet")

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config

try:
    from ultralytics import SAM
except ImportError:
    print("ERROR: pip install ultralytics")
    sys.exit(1)


def get_spice_mask(model, img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Prompt SAM with image center point — spice is almost always centered.
    Returns boolean mask (H×W) or None on failure.
    """
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    results = model(img_rgb, points=[[cx, cy]], labels=[1], verbose=False)

    if not results or results[0].masks is None:
        return None

    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    if len(masks) == 0:
        return None

    # Pick the mask closest to center (largest overlap with center region)
    center_box = np.zeros((h, w), dtype=bool)
    center_box[h//4:3*h//4, w//4:3*w//4] = True
    scores = [np.logical_and(m.astype(bool), center_box).sum() for m in masks]
    best = masks[np.argmax(scores)].astype(bool)

    # Reject if mask is too small or too large (likely background grab)
    ratio = best.sum() / (h * w)
    if ratio < 0.05 or ratio > 0.95:
        return None

    return best


def remove_background(img_rgb: np.ndarray, mask: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    out = img_rgb.copy()
    out[~mask] = np.array(bg, dtype=np.uint8)
    return out


def process_class(model, src_cls: Path, dst_cls: Path) -> list:
    dst_cls.mkdir(parents=True, exist_ok=True)
    files   = [f for f in src_cls.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    failed  = []

    for fp in tqdm(files, desc=src_cls.name, ncols=80):
        dst_fp = dst_cls / fp.name
        if dst_fp.exists():
            continue

        try:
            img = np.array(Image.open(fp).convert("RGB"))
            mask = get_spice_mask(model, img)

            if mask is not None:
                result = remove_background(img, mask)
            else:
                result = img
                failed.append(str(fp))

            Image.fromarray(result).save(dst_fp, quality=95)

        except Exception as e:
            failed.append(f"{fp}: {e}")
            shutil.copy2(fp, dst_fp)

    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", default=None, help="Process single class only (for testing)")
    parser.add_argument("--model", default="mobile_sam.pt", help="SAM model: mobile_sam.pt or sam_b.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model : {args.model}  (auto-downloads on first use)")

    print("\nLoading SAM model...")
    model = SAM(args.model)
    print("SAM ready.\n")

    src_dir = config.DATA_DIR
    dst_dir = config._BASE / "Spice_Spectrum_SAM"

    classes = [args.cls] if args.cls else config.CLASSES
    print(f"Source      : {src_dir}")
    print(f"Destination : {dst_dir}")
    print(f"Classes     : {classes}\n")

    t0 = time.time()
    all_failed = []

    for cls in classes:
        failed = process_class(model, src_dir / cls, dst_dir / cls)
        all_failed.extend(failed)

    elapsed = time.time() - t0
    total = sum(len(list((dst_dir / c).iterdir())) for c in classes if (dst_dir / c).exists())

    print(f"\nDone in {elapsed/60:.1f} min | {total} images saved → {dst_dir}")

    if all_failed:
        print(f"Fallback (original copied): {len(all_failed)} images")
    else:
        print("All images segmented successfully.")


if __name__ == "__main__":
    main()
