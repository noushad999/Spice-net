# src/trainer.py — 3-Phase Training Loop

## Overview

`PhaseTrainer` implements the three-phase curriculum training strategy.
Each phase has a different objective, frozen/unfrozen set of parameters,
and loss function. The phases must run in order: 1 → 2 → 3.

---

## Class: `PhaseTrainer`

```python
class PhaseTrainer:
    def __init__(self, model, device, ckpt_dir):
        self.model    = model      # SpiceFusionNet instance
        self.device   = device     # torch.device("cuda" or "cpu")
        self.ckpt_dir = ckpt_dir   # Path to outputs/checkpoints/
```

---

## Helper: `_make_scheduler(optimizer, warmup, total, min_lr)`

Creates a two-stage learning rate schedule:

```python
SequentialLR([
    LinearLR(start_factor=0.001, end_factor=1.0, total_iters=warmup),
    CosineAnnealingLR(T_max=total-warmup, eta_min=min_lr),
], milestones=[warmup])
```

**Stage 1 — Linear Warmup (3 epochs):**
LR starts at `0.001 × LR` and rises linearly to full LR.
Prevents large gradient updates at the very start when the model is random.

**Stage 2 — Cosine Annealing (27 epochs):**
LR decreases following a cosine curve from full LR down to `min_lr=1e-6`.
Allows precise fine-tuning as training converges.

---

## Phase 1: `phase1(train_loader, val_loader)` → history dict

### Goal
Establish a strong image-based feature extractor using EfficientNet-B4 alone.
No hand-crafted features yet. Just learn to recognize spices from pixels.

### Setup
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = _make_scheduler(optimizer, warmup=3, total=30, min_lr=1e-6)
```

All parameters are trainable. The full EfficientNet-B4 is fine-tuned.

### Training Loop (per epoch)

**Forward pass:**
```python
for imgs, tex, col, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    logits = model.forward_image(imgs)    # ignore tex, col
    loss   = criterion(logits, labels)
```

`tex` and `col` are loaded but ignored — they are zero tensors in Phase 1
(since `get_dataloaders(multimodal=False)`).

**Gradient update:**
```python
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # prevent exploding gradients
optimizer.step()
```

**Evaluation (after each epoch):**
```python
val_loss, val_acc = self._eval_image(model, val_loader, criterion)
```
Runs on validation set, no gradient computation (`@torch.no_grad()`).

### Early Stopping
```python
if val_acc > best_acc:
    best_acc = val_acc
    save_checkpoint("p1_best.pth", ...)   # save best weights
    patience = 0
else:
    patience += 1
    if patience >= 8:
        print("Early stop")
        break
```
If validation accuracy doesn't improve for 8 consecutive epochs, training stops.
This prevents overfitting and saves compute.

### Checkpoints Saved
- `p1_best.pth` — highest validation accuracy weights (used as input to Phase 2)
- `p1_last.pth` — final epoch weights (for reference)

### History Recorded
```python
history = {
    "train_loss": [...],   # per-epoch training loss
    "train_acc":  [...],   # per-epoch training accuracy
    "val_loss":   [...],   # per-epoch validation loss
    "val_acc":    [...],   # per-epoch validation accuracy
    "lr":         [...],   # learning rate at each epoch
}
```
Returned and saved to `p1_training_curves.png`.

---

## Phase 2: `phase2(train_loader)` → None

### Goal
Refine the CNN embedding space using Supervised Contrastive Learning.
Hard-negative pairs (coriander vs cumin) should be pushed apart.

### Load Phase 1 Checkpoint
```python
ckpt = torch.load("p1_best.pth")
self.model.load_state_dict(ckpt["model_state"])
```
Phase 2 starts from the best Phase 1 weights, not random initialization.

### Freeze Most Parameters
```python
for name, p in model.named_parameters():
    p.requires_grad = any(k in name for k in ("backbone", "proj_head"))
```

| Component | Trainable? |
|---|---|
| `backbone` (EfficientNet-B4) | YES |
| `proj_head` | YES |
| `tex_branch`, `col_branch` | Frozen |
| `img_head`, `fusion_head` | Frozen |

Only backbone + projection head are updated. The texture/color branches and
classifiers are frozen to avoid disrupting Phase 1's learned classification weights.

Trainable parameters: ~20.3M (backbone 19.3M + proj_head ~1M).

### Training Loop (per epoch)
```python
for imgs, tex, col, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    proj = model.forward_contrastive(imgs)   # (B, 128) L2-normalized
    loss = supcon(proj, labels)
```

No validation step — SupCon loss has no direct accuracy metric.
The purpose is purely to reshape the embedding geometry.

### After Phase 2
Re-enable all parameters:
```python
for p in model.parameters():
    p.requires_grad = True
```

### Checkpoint Saved
- `p2_last.pth` — final epoch weights (no "best" since no val metric)

---

## Phase 3: `phase3(train_loader, val_loader)` → history dict

### Goal
Train all three branches end-to-end with combined CE+SupCon loss.
The AttentionFusion module learns to weight the three branches.

### Load Phase 2 Checkpoint
```python
ckpt = torch.load("p2_last.pth")
self.model.load_state_dict(ckpt["model_state"])
```

### Setup
```python
loss_fn   = CombinedLoss(alpha=0.5, label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
```

LR is lower (1e-5) because all parameters are pretrained — we only need fine adjustment.

### Training Loop (per epoch)
```python
for imgs, tex, col, labels in train_loader:
    imgs, tex, col, labels = [t.to(device) for t in (imgs, tex, col, labels)]
    logits, proj = model.forward_fusion(imgs, tex, col)
    loss = loss_fn(logits, proj, labels)   # 0.5*CE + 0.5*SupCon
```

All three branches are now active. tex and col are real features
(multimodal=True dataloaders required for Phase 3).

### Validation
```python
val_loss, val_acc = self._eval_fusion(model, val_loader)
```
Uses only CE loss for validation (SupCon has no classification metric).

### Checkpoints Saved
- `best.pth` — highest validation accuracy (primary final model)
- `last.pth` — final epoch weights

---

## Helper: `_eval_image(model, loader, criterion)`

```python
@torch.no_grad()
def _eval_image(self, model, loader, criterion):
    model.eval()
    for imgs, tex, col, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model.forward_image(imgs)
        # accumulate loss and correct count
    return total_loss / n, total_correct / n
```

Used in Phase 1 for validation.

## Helper: `_eval_fusion(model, loader)`

```python
@torch.no_grad()
def _eval_fusion(self, model, loader):
    model.eval()
    for imgs, tex, col, labels in loader:
        logits, _ = model.forward_fusion(imgs, tex, col)
        # CE loss only (ignore SupCon for validation)
    return total_loss / n, total_correct / n
```

Used in Phase 3 for validation. Returns only CE-based loss and accuracy.

---

## W&B Logging

```python
def _log(d: dict):
    if config.USE_WANDB and _WANDB:
        _wandb.log(d)
```

Optional integration. Logs metrics like `p1/train_loss`, `p2/supcon_loss`, `p3/val_acc`
to Weights & Biases if `USE_WANDB=True` in `config.py`.

---

## Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied in all three phases. If the gradient norm exceeds 1.0, all gradients are
scaled down proportionally. Prevents exploding gradients, which can cause NaN losses,
especially when training EfficientNet-B4 end-to-end.
