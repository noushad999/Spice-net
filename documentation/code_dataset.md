# src/dataset.py — Data Loading, Splits, and Augmentation

## Overview

This module handles everything related to raw data:
1. Defining image augmentation pipelines (train vs val/test)
2. Splitting the dataset into train / val / test
3. Loading images and extracting hand-crafted features per sample
4. Wrapping everything into PyTorch DataLoaders

---

## Augmentation Pipelines

### `get_train_transform()` — used during training

```python
def get_train_transform(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,32),
                        hole_width_range=(8,32), fill=0, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
```

Uses [Albumentations](https://albumentations.ai/) (faster than torchvision transforms,
supports numpy arrays directly).

**Why each augmentation:**

| Transform | Reasoning |
|---|---|
| `RandomResizedCrop` | Images may be cropped differently in real use; teaches scale invariance |
| `HorizontalFlip` + `VerticalFlip` | Spice appearance doesn't change with orientation |
| `Rotate(30°)` | Handles tilted camera angles in real applications |
| `ColorJitter` | Handles different lighting conditions and camera sensors |
| `GaussianBlur` / `MotionBlur` | Robustness to camera blur and motion |
| `GaussNoise` | Robustness to sensor noise |
| `CoarseDropout` | Simulates partial occlusion (e.g., overlapping spices) |
| `Normalize` | Shifts pixel distribution to match ImageNet pretrain statistics |

### `get_val_transform()` — used for validation and test

```python
def get_val_transform(img_size=224):
    resize = int(224 * 256 / 224)   # = 256
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(...),
        ToTensorV2(),
    ])
```

No random augmentation — deterministic resize+crop+normalize only.
This matches the standard ImageNet evaluation protocol.

---

## Dataset Splitting: `build_splits()`

```python
def build_splits(data_dir=config.DATA_DIR):
    paths, labels = _collect(data_dir, config.CLASSES)

    # Step 1: split off 30% for val+test (stratified by class)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42
    )

    # Step 2: from the 30%, split val=10% and test=20%
    val_frac = 0.10 / 0.30   # = 0.333
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=1 - val_frac, stratify=y_tmp, random_state=42
    )

    return x_tr, y_tr, x_val, y_val, x_te, y_te
```

**Key design decisions:**

- `stratify=labels` — each split has the same class distribution (balanced splits)
- `random_state=42` — fully deterministic; same images always go to same split
- Returns file paths (strings), not loaded images (memory-efficient)

**Internal helper `_collect()`:**
Walks each class folder, collects all `.jpg/.jpeg/.png` files, assigns integer labels
based on the class index in `config.CLASSES`.

---

## Class: `SpiceDataset`

```python
class SpiceDataset(Dataset):
    def __init__(self, paths, labels, transform=None, multimodal=False):
```

PyTorch `Dataset` — the `__getitem__` method defines what one sample returns.

### `__getitem__(idx)` step by step:

```python
def __getitem__(self, idx):
    # 1. Load image as RGB numpy array
    img_np = np.array(Image.open(self.paths[idx]).convert("RGB"))

    # 2. Extract hand-crafted features BEFORE augmentation (on original image)
    if self.multimodal:
        tex_np, col_np = _extract_all(img_np)
        tex = torch.from_numpy(tex_np)
        col = torch.from_numpy(col_np)
    else:
        tex = torch.zeros(config.TEX_INPUT_DIM)   # zero placeholders
        col = torch.zeros(config.COL_INPUT_DIM)

    # 3. Apply augmentation to image
    if self.transform:
        img_np = self.transform(image=img_np)["image"]   # Albumentations API

    # 4. Return tuple
    return img_np, tex, col, self.labels[idx]
```

**Critical order:** Hand-crafted features are extracted **before** augmentation.
This is intentional — LBP/GLCM/HSV should describe the true spice appearance,
not a randomly augmented/corrupted version of it.

### Multimodal flag:
- `multimodal=False` (Phase 1, 2, baselines): returns zero tensors for tex/col.
  The trainer ignores these zeros.
- `multimodal=True` (Phase 3): returns real texture and color features.

### Returns: `(img_tensor, tex_tensor, col_tensor, label_int)`
- `img_tensor`: `(3, 224, 224)` float32 — normalized image
- `tex_tensor`: `(58,)` float32 — texture features (or zeros)
- `col_tensor`: `(100,)` float32 — color features (or zeros)
- `label_int`: int — class index 0–10

---

## `get_dataloaders()`

```python
def get_dataloaders(multimodal=False, data_dir=config.DATA_DIR,
                    batch_size=32, num_workers=4):
```

Convenience function that builds all three DataLoaders at once.

```python
tr_ds  = SpiceDataset(x_tr,  y_tr,  get_train_transform(), multimodal)
val_ds = SpiceDataset(x_val, y_val, get_val_transform(),   multimodal)
te_ds  = SpiceDataset(x_te,  y_te,  get_val_transform(),   multimodal)

# pin_memory=True for faster GPU transfer
mk = dict(batch_size=32, num_workers=4, pin_memory=True)

return (
    DataLoader(tr_ds,  shuffle=True,  drop_last=True, **mk),  # train
    DataLoader(val_ds, shuffle=False, **mk),                   # val
    DataLoader(te_ds,  shuffle=False, **mk),                   # test
    x_te, y_te,   # raw file paths + labels (needed for Grad-CAM)
)
```

`drop_last=True` on train loader: SupCon loss requires multiple samples per class
in each batch. Dropping the last (potentially small) batch avoids edge cases where
a class might appear only once.

`pin_memory=True`: Pre-allocates tensors in pinned CPU memory for faster transfer
to GPU via DMA (Direct Memory Access).
