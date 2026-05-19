# src/model.py — SpiceFusionNet Architecture

## Overview

`SpiceFusionNet` is the core model. It combines three information streams:
1. **CNN Branch** — deep image features from EfficientNet-B4
2. **Texture Branch** — hand-crafted LBP+GLCM features processed by an MLP
3. **Color Branch** — hand-crafted HSV histogram features processed by an MLP

These three are merged by `AttentionFusion` with learned weights, then classified.

---

## Class: `_MLP`

```python
class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, drop=0.3):
```

A reusable two-layer MLP with this structure:

```
Linear(in → hidden) → BatchNorm → ReLU → Dropout(drop) → Linear(hidden → out) → BatchNorm → ReLU
```

Used twice:
- **Texture branch**: `_MLP(58, 128, 256)` — maps 58-d texture vector to 256-d
- **Color branch**: `_MLP(100, 64, 128)` — maps 100-d color vector to 128-d

Why BatchNorm? It stabilizes training and normalizes the hand-crafted feature scale,
which can vary wildly (GLCM values differ in magnitude from LBP histogram values).

---

## Class: `AttentionFusion`

```python
class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim, tex_dim, col_dim):
        self.gate = nn.Sequential(
            nn.Linear(2176, 3),
            nn.Softmax(dim=1),
        )
```

**How it works:**

1. Concatenate all three branch outputs → 2176-d vector
2. Pass through a linear layer → 3 raw scores (one per branch)
3. Softmax → 3 weights that sum to 1.0 (α_img, α_tex, α_col)
4. Multiply each branch's features by its weight, then re-concatenate

```python
def forward(self, f_cnn, f_tex, f_col):
    cat = torch.cat([f_cnn, f_tex, f_col], dim=1)    # (B, 2176)
    g   = self.gate(cat)                               # (B, 3) — sums to 1
    out = torch.cat([
        g[:, 0:1] * f_cnn,
        g[:, 1:2] * f_tex,
        g[:, 2:3] * f_col,
    ], dim=1)
    return out    # (B, 2176) — weighted features
```

**Why not simple concatenation?**
Different classes benefit from different features. Turmeric (bright yellow powder) relies
more on color; saffron threads rely more on texture. The gate learns these class-specific
preferences during Phase 3.

The output is still 2176-d (not 3-d) — the gating is multiplicative, not a bottleneck.

---

## Class: `SpiceFusionNet`

```python
class SpiceFusionNet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, ...):
```

### Components Built in `__init__`:

#### 1. CNN Backbone
```python
self.backbone = timm.create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=0,        # removes the original classification head
    global_pool="avg",    # global average pooling → 1792-d vector
    drop_rate=0.4,
)
```
`num_classes=0` removes EfficientNet's original 1000-class head. The backbone outputs
a 1792-d feature vector per image.

#### 2. Texture Branch
```python
self.tex_branch = _MLP(58, 128, 256, drop=0.3)
```
Input: 58-d (LBP 10-d + GLCM 48-d) → Output: 256-d

#### 3. Color Branch
```python
self.col_branch = _MLP(100, 64, 128, drop=0.3)
```
Input: 100-d HSV histogram → Output: 128-d

#### 4. Attention Fusion
```python
self.fusion = AttentionFusion(1792, 256, 128)
```
Merges three branches with learned gates.

#### 5. Projection Head (Phase 2 only)
```python
self.proj_head = nn.Sequential(
    nn.Linear(1792, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
)
```
Used exclusively in Phase 2 for SupCon loss. Projects CNN features to a
128-d embedding space where contrastive learning is applied.
L2 normalization is applied in `forward_contrastive`, not here.

#### 6. Image-Only Classifier (Phase 1)
```python
self.img_head = nn.Sequential(
    nn.Linear(1792, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 11),
)
```
Takes the 1792-d CNN features and produces 11 class logits.
Used only in Phase 1.

#### 7. Fusion Classifier (Phase 3)
```python
self.fusion_head = nn.Sequential(
    nn.Linear(2176, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 11),
)
```
Takes the 2176-d fused features (after AttentionFusion) and produces 11 logits.

---

## Three Forward Modes

### `forward_image(x)` — Phase 1
```python
def forward_image(self, x):
    return self.img_head(self.backbone(x))
```
Flow: `image → backbone → 1792-d → img_head → 11 logits`
Loss: CrossEntropy

### `forward_contrastive(x)` — Phase 2
```python
def forward_contrastive(self, x):
    feats = self.backbone(x)
    proj  = self.proj_head(feats)
    return F.normalize(proj, dim=1)    # L2 normalize → unit sphere
```
Flow: `image → backbone → 1792-d → proj_head → 128-d → L2 normalize`
Loss: SupConLoss (cosine similarity in unit sphere)

L2 normalization maps all projections to a unit hypersphere. This is required by
SupCon loss, which uses cosine similarity between pairs.

### `forward_fusion(x, tex, col)` — Phase 3
```python
def forward_fusion(self, x, tex, col):
    f_cnn  = self.backbone(x)          # 1792-d
    f_tex  = self.tex_branch(tex)      # 256-d
    f_col  = self.col_branch(col)      # 128-d
    fused  = self.fusion(f_cnn, f_tex, f_col)  # 2176-d (gated)
    logits = self.fusion_head(fused)   # 11 logits
    proj   = F.normalize(self.proj_head(f_cnn), dim=1)  # 128-d for SupCon
    return logits, proj
```
Returns both logits (for CE loss) and projection (for SupCon loss) simultaneously.

### `forward(x, tex=None, col=None)` — convenience wrapper
```python
def forward(self, x, tex=None, col=None):
    if tex is not None and col is not None:
        logits, _ = self.forward_fusion(x, tex, col)
        return logits
    return self.forward_image(x)
```
Auto-selects mode based on whether hand-crafted features are passed.
Used by baseline evaluation code (which just calls `model(imgs)`).

---

## Checkpoint Functions

### `save_checkpoint(path, model, optimizer, epoch, best_val_acc, history)`
Saves:
- `model_state` — all parameter weights
- `optimizer_state` — optimizer state (for resuming training)
- `epoch` — which epoch this was saved at
- `best_val_acc` — for reference
- `history` — loss/acc curves dict

### `load_checkpoint(path, device)`
Rebuilds a `SpiceFusionNet` and loads saved weights.
Returns: `(model, epoch, best_val_acc, history)`

---

## Parameter Count

| Component | Parameters |
|---|---|
| EfficientNet-B4 backbone | ~19.3 M |
| tex_branch MLP | ~41 K |
| col_branch MLP | ~15 K |
| AttentionFusion + fusion_head | ~1.2 M |
| proj_head + img_head | ~1.0 M |
| **Total** | **~21.6 M** |

EfficientNet-B4 dominates at ~89% of total parameters.
