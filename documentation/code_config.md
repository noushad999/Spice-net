# config.py — Central Hyperparameter Registry

## Purpose

All hyperparameters, paths, and constants for the entire project live here.
No training script hardcodes a number — every tunable value is imported from `config`.
To change anything (learning rate, image size, batch size), edit only this file.

---

## Path Resolution

```python
def _resolve_base() -> Path:
    wsl = Path("/mnt/d/SpiceNet")
    win = Path("D:/SpiceNet")
    return wsl if wsl.exists() else win
```

The project runs on both Windows (`D:/SpiceNet`) and WSL (`/mnt/d/SpiceNet`).
`_resolve_base()` auto-detects the environment at import time so paths always resolve correctly.

---

## Directory Layout

| Variable | Path | Purpose |
|---|---|---|
| `DATA_DIR` | `Spice_Spectrum/` | Raw dataset |
| `DATA_DIR_SAM` | `Spice_Spectrum_SAM/` | SAM background-removed dataset (Ablation A5) |
| `OUTPUT_DIR` | `outputs/` | All generated outputs (curves, confusion matrices) |
| `CHECKPOINT_DIR` | `outputs/checkpoints/` | Saved model weights (.pth files) |
| `LOG_DIR` | `outputs/logs/` | Training logs |

---

## Class Definitions

```python
CLASSES = [
    "black pepper", "cardamom", "cinnamon", "cloves", "coriander",
    "cumin", "ginger", "nutmeg", "paprika", "saffron", "turmeric",
]
NUM_CLASSES = 11
```

The index of a class in this list is its integer label throughout the codebase.
Example: `"coriander"` is class `4`, `"cumin"` is class `5`.

---

## Hard-Negative Pairs

```python
HARD_NEG_PAIRS = [
    (4, 5),   # coriander ↔ cumin      (small beige seeds)
    (8, 10),  # paprika   ↔ turmeric   (orange-yellow powder)
    (0, 3),   # black pepper ↔ cloves  (dark spheroidal)
    (2, 7),   # cinnamon  ↔ nutmeg     (brown powder)
]
```

These pairs are visually confusable. Supervised Contrastive Loss (Phase 2) specifically
pushes these pairs apart in embedding space.

---

## Data Splits

```python
TRAIN_RATIO  = 0.70   # 7,700 images
VAL_RATIO    = 0.10   # 1,100 images
TEST_RATIO   = 0.20   # 2,200 images
RANDOM_SEED  = 42
```

`RANDOM_SEED=42` makes the split deterministic — same images land in the same split
across all runs and all machines. The test set is strictly held out.

---

## Image Resolution

```python
IMG_SIZE = 224    # CNN input (after resize+crop)
IMG_FULL = 512    # Hand-crafted feature extraction resolution
IMG_MEAN = (0.485, 0.456, 0.406)   # ImageNet mean
IMG_STD  = (0.229, 0.224, 0.225)   # ImageNet std
```

Two resolutions are used deliberately:
- `224` — EfficientNet-B4's standard input size
- `512` — Higher resolution for LBP/GLCM/HSV to capture fine texture/color detail

---

## Model Architecture Dimensions

```python
BACKBONE  = "efficientnet_b4"  # timm model name
CNN_DIM   = 1792   # EfficientNet-B4 global avg pool output
TEX_DIM   = 256    # Texture MLP output
COL_DIM   = 128    # Color MLP output
PROJ_DIM  = 128    # SupCon projection head output
DROP_RATE = 0.4    # Dropout in classifiers
```

The fused representation dimension = `CNN_DIM + TEX_DIM + COL_DIM = 2176`.

---

## Hand-Crafted Feature Dimensions

### Texture (LBP + GLCM)

```python
LBP_P           = 8              # Number of LBP neighbor points
LBP_R           = 1              # LBP radius
GLCM_DISTANCES  = [1, 2]         # Pixel distances for GLCM
GLCM_ANGLES_DEG = [0, 45, 90, 135]  # Directions (4 angles)
GLCM_LEVELS     = 64             # Gray-level quantization

# LBP output: LBP_P + 2 = 10 bins
# GLCM output: 6 props × 2 distances × 4 angles = 48
TEX_INPUT_DIM = 10 + 48 = 58
```

### Color (HSV Histogram)

```python
HSV_H_BINS = 36    # Hue (0-180° range → 36 bins of 5° each)
HSV_S_BINS = 32    # Saturation
HSV_V_BINS = 32    # Value (brightness)
COL_INPUT_DIM = 36 + 32 + 32 = 100
```

---

## 3-Phase Training Hyperparameters

### Phase 1 — Backbone Pre-Training
```python
P1_EPOCHS       = 30
P1_LR           = 1e-4       # AdamW learning rate
P1_WARMUP       = 3          # Linear warmup epochs
P1_WEIGHT_DECAY = 1e-4
P1_LABEL_SMOOTH = 0.1        # CrossEntropy label smoothing
P1_MIN_LR       = 1e-6       # CosineAnnealing floor
```

### Phase 2 — Contrastive Fine-Tuning
```python
P2_EPOCHS      = 10
P2_LR          = 5e-5        # Lower LR (fine-tuning)
P2_TEMPERATURE = 0.07        # SupCon temperature τ
```

Lower temperature → sharper contrast between positive and negative pairs.
`0.07` is the standard value from the original SupCon paper (Khosla et al., 2020).

### Phase 3 — Full Fusion
```python
P3_EPOCHS       = 10
P3_LR           = 1e-5       # Even lower (all params active)
P3_WEIGHT_DECAY = 1e-4
P3_LABEL_SMOOTH = 0.1
P3_ALPHA        = 0.5        # CE weight in: alpha*CE + (1-alpha)*SupCon
```

---

## Shared Training Settings

```python
BATCH_SIZE  = 32
NUM_WORKERS = 4       # DataLoader parallel workers
GRAD_CLIP   = 1.0    # Max gradient norm (prevents exploding gradients)
PATIENCE    = 8      # Early stopping patience (epochs without improvement)
```

---

## Augmentation Parameters

```python
AUG_ROTATION   = 30           # Random rotate ±30°
AUG_SCALE      = (0.7, 1.0)  # RandomResizedCrop scale range
AUG_BRIGHTNESS = 0.3          # ColorJitter brightness
AUG_CONTRAST   = 0.3
AUG_SATURATION = 0.3
AUG_HUE        = 0.1
```

---

## W&B (Weights & Biases) Integration

```python
USE_WANDB      = False          # Set True to enable experiment tracking
WANDB_PROJECT  = "spicenet"
```

When `USE_WANDB=True`, all training metrics are logged to wandb for visualization.
