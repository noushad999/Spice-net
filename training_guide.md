# Training Guide

## Overview

SpiceNet uses a **3-phase curriculum** that progressively adds complexity:

| Phase | Name | Duration | Loss | What trains |
|---|---|---|---|---|
| 1 | Backbone pre-training | 30 epochs | CE + label smoothing | CNN backbone + `img_head` |
| 2 | Contrastive fine-tuning | 10 epochs | SupCon | Backbone + `proj_head` (others frozen) |
| 3 | Full fusion | 10 epochs | 0.5×CE + 0.5×SupCon | Everything (all branches) |

---

## Running Training

### Full pipeline (all 3 phases, recommended)
```bash
python train.py
```

### Single phase
```bash
python train.py --phase 1    # Phase 1 only
python train.py --phase 2    # Phase 2 only (requires p1_best.pth)
python train.py --phase 3    # Phase 3 only (requires p2_last.pth)
```

### With SAM-preprocessed data (background removal ablation)
```bash
python train.py --data_dir Spice_Spectrum_SAM
```

### Enable multimodal during Phase 1 (non-standard)
```bash
python train.py --phase 1 --multimodal
```

> Phase 3 always enables multimodal automatically regardless of the flag.

---

## Phase 1 — Backbone Pre-Training

**Goal:** Establish a strong image-based feature extractor before adding hand-crafted features.

```
Loss      : CrossEntropyLoss(label_smoothing=0.1)
Optimizer : AdamW(lr=1e-4, weight_decay=1e-4)
Scheduler : LinearWarmup(3 epochs) → CosineAnnealingLR(T_max=27)
Epochs    : 30
Early stop: patience=8 (monitors val accuracy)
Batch size: 32
```

**Checkpoints saved:**
- `outputs/checkpoints/p1_best.pth` — best validation accuracy
- `outputs/checkpoints/p1_last.pth` — final epoch

**Outputs:**
- `outputs/p1_training_curves.png` — loss, accuracy, LR over epochs

---

## Phase 2 — Contrastive Fine-Tuning

**Goal:** Shape the embedding space so that same-class spices cluster tightly and hard-negative pairs
(e.g., coriander vs cumin) are pushed apart.

```
Loss      : SupConLoss(temperature=0.07)
Optimizer : AdamW(lr=5e-5, weight_decay=1e-4)
Epochs    : 10
Frozen    : texture_branch, color_branch, img_head, fusion_head
Trainable : backbone, proj_head
```

**Input:** Phase 1 checkpoint (`p1_best.pth`)

**Checkpoints saved:**
- `outputs/checkpoints/p2_last.pth` — final epoch (no best; SupCon has no val metric)

> SupCon does not use labels for classification, so validation accuracy is not applicable.
> The purpose is purely embedding quality improvement.

---

## Phase 3 — Full Fusion End-to-End

**Goal:** Combine all three branches and train the complete multi-modal classifier.

```
Loss      : CombinedLoss(alpha=0.5) = 0.5×CE + 0.5×SupCon
Optimizer : AdamW(lr=1e-5, weight_decay=1e-4)
Scheduler : CosineAnnealingLR(T_max=10)
Epochs    : 10
Trainable : all parameters
```

**Input:** Phase 2 checkpoint (`p2_last.pth`)

**Checkpoints saved:**
- `outputs/checkpoints/best.pth` — best validation accuracy (primary model)
- `outputs/checkpoints/last.pth` — final epoch

**Outputs:**
- `outputs/p3_training_curves.png` — loss, accuracy, LR over epochs

---

## Hyperparameter Reference

All hyperparameters are centralized in `config.py`.

### Data
| Parameter | Value | Description |
|---|---|---|
| `IMAGE_SIZE` | 224 | CNN input resolution |
| `FULL_SIZE` | 512 | Feature extraction resolution |
| `TRAIN_RATIO` | 0.70 | Training split fraction |
| `VAL_RATIO` | 0.10 | Validation split fraction |
| `TEST_RATIO` | 0.20 | Test split fraction |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `NUM_WORKERS` | 4 | DataLoader workers |

### Phase 1
| Parameter | Value |
|---|---|
| `P1_EPOCHS` | 30 |
| `P1_LR` | 1e-4 |
| `P1_WEIGHT_DECAY` | 1e-4 |
| `P1_BATCH_SIZE` | 32 |
| `P1_LABEL_SMOOTHING` | 0.1 |
| `P1_PATIENCE` | 8 |
| `P1_WARMUP_EPOCHS` | 3 |

### Phase 2
| Parameter | Value |
|---|---|
| `P2_EPOCHS` | 10 |
| `P2_LR` | 5e-5 |
| `P2_TEMPERATURE` | 0.07 |

### Phase 3
| Parameter | Value |
|---|---|
| `P3_EPOCHS` | 10 |
| `P3_LR` | 1e-5 |
| `P3_ALPHA` | 0.5 |

### Model
| Parameter | Value | Description |
|---|---|---|
| `BACKBONE` | `efficientnet_b4` | timm model name |
| `NUM_CLASSES` | 11 | Number of spice classes |
| `TEXTURE_DIM` | 256 | Texture branch output |
| `COLOR_DIM` | 128 | Color branch output |
| `PROJ_DIM` | 128 | Contrastive projection dim |

---

## Checkpoint Format

Checkpoints are PyTorch `state_dict` saved with metadata:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "best_val_acc": float,
    "config": {
        "num_classes": 11,
        "backbone": "efficientnet_b4",
        "texture_dim": 256,
        "color_dim": 128,
        "proj_dim": 128,
    }
}
```

Load a checkpoint:
```python
from src.model import SpiceFusionNet, load_checkpoint

model = SpiceFusionNet()
model, meta = load_checkpoint(model, "outputs/checkpoints/best.pth")
print(f"Loaded epoch {meta['epoch']}, val_acc={meta['best_val_acc']:.4f}")
```

---

## Data Augmentation

Training augmentation is defined in `src/dataset.py → get_train_transform()`.

| Transform | Parameters | Purpose |
|---|---|---|
| `RandomResizedCrop` | scale=(0.7,1.0) | Scale invariance |
| `HorizontalFlip` | p=0.5 | Orientation invariance |
| `VerticalFlip` | p=0.3 | Orientation invariance |
| `Rotate` | limit=30° | Rotation invariance |
| `ColorJitter` | brightness, contrast, saturation | Lighting robustness |
| `GaussianBlur` | p=0.2 | Blur robustness |
| `GaussianNoise` | p=0.2 | Noise robustness |
| `CoarseDropout` | p=0.3 | Occlusion robustness |
| `Normalize` | ImageNet mean/std | Distribution alignment |

Validation/test only applies `Resize(256) → CenterCrop(224) → Normalize`.

> Hand-crafted features (LBP, GLCM, HSV) are extracted from the **original 512×512 image**
> before augmentation, so they represent the true signal without stochastic corruption.
