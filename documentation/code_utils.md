# src/utils.py — Shared Utilities

## Overview

Utility functions shared across training, evaluation, and inference scripts.
These avoid code duplication and ensure consistent metric computation and visualization.

---

## `set_seed(seed=42)`

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Sets all random number generators to the same seed for reproducibility.

**Why four different seeds?**
- `random.seed` — Python's built-in random module (used in `build_test_folder.py`)
- `np.random.seed` — NumPy (used in feature extraction, dataset operations)
- `torch.manual_seed` — PyTorch CPU operations
- `cudnn.deterministic=True` — makes CUDA ops deterministic (slight performance cost)

`cudnn.benchmark=False`: When True, cuDNN auto-selects the fastest algorithm (non-deterministic).
Setting False ensures identical results across runs.

Called at the start of every script (`train.py`, `evaluate.py`, `run_baselines.py`, etc.).

---

## `topk_accuracy(output, target, k=5)` → float

```python
def topk_accuracy(output, target, k=5):
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)  # (B, k) top-k indices
    correct = pred.eq(target.view(-1, 1).expand_as(pred))        # is true label in top-k?
    return correct.any(dim=1).float().sum().item() / batch
```

Computes Top-K accuracy: prediction is correct if the true label appears
anywhere in the top-k predicted classes.

Example: `topk_accuracy(logits, labels, k=5)` → 0.9995 means 99.95% of the time,
the correct class is in the model's top-5 predictions.

Used in `evaluate.py` with `k=min(5, 11)` to compute Top-5 accuracy.

---

## `measure_inference_time(model, loader, device, n_batches=20)` → float (ms/img)

```python
def measure_inference_time(model, loader, device, n_batches=20):
    forward_fn = getattr(model, "forward_image", model)   # handles SpiceFusionNet and plain timm models
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        imgs = batch[0].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()    # flush GPU queue before timing
        t0 = time.perf_counter()
        _ = forward_fn(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()    # wait for GPU to finish
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed / imgs.size(0))  # per-image time
    return float(np.mean(times))
```

**Why `cuda.synchronize()`?**
GPU operations are asynchronous — the CPU doesn't wait for GPU to finish.
Without synchronization, `time.perf_counter()` would measure near-zero CPU dispatch time,
not actual GPU compute time.

**Why average over 20 batches?**
Single batch timing has high variance (cache effects, pipeline warmup).
Averaging gives a stable estimate.

**`getattr(model, "forward_image", model)`:**
SpiceFusionNet has `forward_image()` (image-only, Phase 1 speed).
Plain timm baselines (ResNet, ViT) don't have this method — falls back to `__call__`.
This makes one function work for all model types.

---

## `plot_training_curves(history, output_dir, prefix="")`

```python
def plot_training_curves(history, output_dir, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # axes[0]: train/val Loss curve
    # axes[1]: train/val Accuracy curve
    # axes[2]: Learning Rate curve (log scale)
    plt.savefig(output_dir / f"{prefix}_training_curves.png", dpi=150)
```

Saves a 3-panel figure:
- **Left**: training and validation loss over epochs
- **Middle**: training and validation accuracy over epochs
- **Right**: learning rate schedule on log scale

Outputs: `outputs/p1_training_curves.png`, `outputs/p3_training_curves.png`

---

## `plot_confusion_matrix(y_true, y_pred, classes, output_dir, prefix="")`

```python
def plot_confusion_matrix(y_true, y_pred, classes, output_dir, prefix=""):
    cm      = confusion_matrix(y_true, y_pred)          # raw counts
    cm_norm = cm / cm.sum(axis=1, keepdims=True)        # row-normalized (per class recall)

    # Two side-by-side subplots: counts and normalized
    sns.heatmap(cm,      ..., fmt="d")     # integer count annotations
    sns.heatmap(cm_norm, ..., fmt=".2f")   # decimal probability annotations
```

Saves two confusion matrices side-by-side:
- **Left**: raw counts — how many images of each class were predicted as what
- **Right**: normalized by true class — shows recall per class (diagonal = recall)

The diagonal of the normalized matrix = per-class recall rate.
Off-diagonal entries reveal which class pairs are confused.

Example output: if row "coriander", column "cumin" has value 0.03,
it means 3% of coriander images were wrongly predicted as cumin.

---

## `compute_and_print_metrics(y_true, y_pred, classes, top5_acc, infer_ms)` → dict

```python
def compute_and_print_metrics(y_true, y_pred, classes, top5_acc=None, infer_ms=None):
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
```

Prints and returns a dictionary with:
- `top1_accuracy` — overall accuracy
- `top5_accuracy` — if provided
- `f1_weighted` — F1 weighted by class frequency (matches accuracy on balanced data)
- `f1_macro` — unweighted average F1 across all classes (better for imbalanced)
- `inference_ms` — inference time if provided

Also prints sklearn's `classification_report` which shows per-class precision/recall/F1.

**F1 macro vs weighted:**
- `macro`: treats all 11 classes equally — good for understanding worst-case class
- `weighted`: weights by class size — since all classes are ~equal size, similar to accuracy here

---

## `save_metrics(metrics, path)`

```python
def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
```

Saves metrics as a formatted JSON file for reproducibility and paper reporting.
Example output: `outputs/fusion_test_metrics.json`
