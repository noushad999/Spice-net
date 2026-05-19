import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

import config

try:
    from src.features import extract_all as _extract_all
    _FEATURES_OK = True
except Exception:
    _FEATURES_OK = False


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=config.AUG_SCALE, ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=config.AUG_ROTATION, p=0.6),
        A.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
            saturation=config.AUG_SATURATION,
            hue=config.AUG_HUE,
            p=0.7,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = config.IMG_SIZE) -> A.Compose:
    resize = int(img_size * 256 / 224)
    return A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


# ── Splits ────────────────────────────────────────────────────────────────────

def _collect(data_dir: Path, classes: List[str]):
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        for fp in sorted((data_dir / cls).iterdir()):
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(fp))
                labels.append(idx)
    return paths, labels


def build_splits(data_dir: Path = config.DATA_DIR):
    paths, labels = _collect(data_dir, config.CLASSES)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=1 - config.TRAIN_RATIO,
        stratify=labels, random_state=config.RANDOM_SEED,
    )
    val_frac = config.VAL_RATIO / (1 - config.TRAIN_RATIO)
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=1 - val_frac,
        stratify=y_tmp, random_state=config.RANDOM_SEED,
    )
    return x_tr, y_tr, x_val, y_val, x_te, y_te


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpiceDataset(Dataset):
    """Unified dataset — returns (image, texture, color, label).
    texture/color are zero tensors if multimodal=False or features unavailable.
    """
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        multimodal: bool = False,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.multimodal = multimodal and _FEATURES_OK

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_np = np.array(Image.open(self.paths[idx]).convert("RGB"))

        # Extract hand-crafted features BEFORE augmentation (on original image)
        if self.multimodal:
            tex_np, col_np = _extract_all(img_np)
            tex = torch.from_numpy(tex_np)
            col = torch.from_numpy(col_np)
        else:
            tex = torch.zeros(config.TEX_INPUT_DIM)
            col = torch.zeros(config.COL_INPUT_DIM)

        if self.transform:
            img_np = self.transform(image=img_np)["image"]

        return img_np, tex, col, self.labels[idx]


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    multimodal: bool = False,
    data_dir: Path = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
):
    x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(data_dir)

    tr_ds  = SpiceDataset(x_tr,  y_tr,  get_train_transform(), multimodal)
    val_ds = SpiceDataset(x_val, y_val, get_val_transform(),   multimodal)
    te_ds  = SpiceDataset(x_te,  y_te,  get_val_transform(),   multimodal)

    mk = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(tr_ds,  shuffle=True,  drop_last=True, **mk),
        DataLoader(val_ds, shuffle=False, **mk),
        DataLoader(te_ds,  shuffle=False, **mk),
        x_te, y_te,
    )
