# SpiceFusionNet — Architecture

## Overview

SpiceFusionNet is a three-branch fusion network built on top of EfficientNet-B4. The three branches
extract complementary information:

1. **CNN Branch** — Deep appearance features from raw pixels (EfficientNet-B4)
2. **Texture Branch** — LBP + GLCM hand-crafted texture descriptors
3. **Color Branch** — HSV histogram color descriptors

An `AttentionFusion` module combines all branches with learned, softmax-normalized weights before
the final classifier.

---

## Architecture Diagram

```
Input Image (224×224×3)
        │
        ├──────────────────────────────────┐
        │                                  │ (512×512, before augmentation)
        ▼                                  ▼
 EfficientNet-B4                  Hand-Crafted Features
 (pretrained, timm)               ┌────────────────────┐
        │                         │ extract_texture()  │
        ▼                         │   LBP  → 10-d      │
  avg pool → 1792-d               │   GLCM → 48-d      │
        │                         │   total: 58-d      │
  ┌─────┴──────┐                  ├────────────────────┤
  │            │                  │ extract_hsv()      │
  ▼            ▼                  │   HSV hist → 100-d │
img_head   proj_head              └────────────────────┘
(Phase 1)  (Phase 2)                       │
  │            │              ┌────────────┴────────────┐
  ▼            ▼              ▼                         ▼
logits×11  embed×128   texture_branch            color_branch
(CE loss)  (L2-norm,   MLP(58→128→256)          MLP(100→64→128)
           SupCon)           │                         │
                             ▼                         ▼
                           256-d                     128-d
                             │                         │
                             └────────────┬────────────┘
                                          │  1792 + 256 + 128 = 2176-d
                                          ▼
                                   AttentionFusion
                                   (learned gates,
                                    softmax weights)
                                          │
                                          ▼
                                   fusion_head
                                   MLP(2176→512→11)
                                   (Phase 3, CE + SupCon)
```

---

## Module Details

### EfficientNet-B4 Backbone (`src/model.py`)

- **Source:** `timm.create_model("efficientnet_b4", pretrained=True)`
- **Output:** 1792-d global average pooled feature vector
- **Training:** Full fine-tuning (all layers) during Phase 1 and Phase 3

### Texture Branch

- **Input:** 58-d vector (LBP 10-d + GLCM 48-d) extracted at 512×512 resolution
- **Architecture:** `_MLP(58, [128], 256)`
  - Linear(58→128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128→256) → BatchNorm → ReLU

### Color Branch

- **Input:** 100-d HSV histogram (36 bins H + 32 bins S + 32 bins V)
- **Architecture:** `_MLP(100, [64], 128)`
  - Linear(100→64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64→128) → BatchNorm → ReLU

### AttentionFusion (`src/model.py`)

Learns per-branch importance weights with a softmax gate:

```python
# Pseudo-code
gates = softmax(Linear(2176, 3))   # α_img, α_tex, α_col
fused = gates[0]*img + gates[1]*tex + gates[2]*col
```

The gate normalizes to sum-to-1, so relative weights are interpretable.

### Projection Head (Phase 2 only)

- **Input:** 1792-d CNN features
- **Architecture:** Linear(1792→512) → ReLU → Linear(512→128) → L2-normalize
- **Purpose:** Supervised contrastive learning embedding space

### Classification Heads

| Head | Phase | Input | Output | Loss |
|---|---|---|---|---|
| `img_head` | 1 | 1792-d CNN | 11 logits | CE + label smoothing |
| `proj_head` | 2 | 1792-d CNN | 128-d L2-norm | SupCon |
| `fusion_head` | 3 | 2176-d fused | 11 logits | CE + SupCon |

---

## Forward Modes

`SpiceFusionNet` exposes three explicit forward methods:

```python
# Phase 1 — image only (classification)
logits = model.forward_image(x)

# Phase 2 — contrastive embedding
embedding = model.forward_contrastive(x)   # L2-normalized, shape (B, 128)

# Phase 3 — full fusion (classification + contrastive)
logits, proj = model.forward_fusion(x, tex, col)

# Auto-select (convenience)
out = model(x)               # → forward_image
out = model(x, tex, col)     # → forward_fusion
```

---

## Feature Dimensions Summary

| Stage | Feature | Dimension |
|---|---|---|
| Raw LBP | uniform histogram | 10 |
| Raw GLCM | 6 props × 2 dist × 4 angles | 48 |
| Texture (concatenated) | LBP + GLCM | 58 |
| HSV Histogram | H(36) + S(32) + V(32) | 100 |
| CNN pool | EfficientNet-B4 avg pool | 1792 |
| Texture branch output | after MLP | 256 |
| Color branch output | after MLP | 128 |
| Fused representation | concatenated | 2176 |
| Projection | SupCon embedding | 128 |
| Final logits | 11-class softmax | 11 |

---

## Parameter Count

| Component | Parameters |
|---|---|
| EfficientNet-B4 backbone | ~19.3 M |
| Texture branch MLP | ~41 K |
| Color branch MLP | ~15 K |
| AttentionFusion + fusion_head | ~1.2 M |
| proj_head + img_head | ~1.0 M |
| **Total** | **~21.6 M** |
