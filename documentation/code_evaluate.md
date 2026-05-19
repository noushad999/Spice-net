# evaluate.py — Test Set Evaluation

## Overview

Runs the trained model on the held-out test split (20% of dataset, never seen during training).
Computes comprehensive metrics and optionally generates Grad-CAM visualizations.

---

## Command-Line Arguments

```bash
python evaluate.py                           # Phase 3 fusion model
python evaluate.py --mode image              # Phase 1 image-only model
python evaluate.py --ckpt p1_best.pth --mode image
python evaluate.py --gradcam                 # Also generate Grad-CAM plots
python evaluate.py --data_dir /path/to/SAM  # Evaluate on SAM-preprocessed data
```

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `outputs/checkpoints/best.pth` | Path to checkpoint |
| `--mode` | `"fusion"` | `"fusion"` or `"image"` |
| `--gradcam` | `False` | Generate Grad-CAM visualization |
| `--data_dir` | `config.DATA_DIR` | Dataset directory |

---

## Execution Flow

```python
# 1. Load model from checkpoint
model, epoch, best_val_acc, _ = load_checkpoint(args.ckpt, device)

# 2. Load test DataLoader
_, _, test_loader, x_te, y_te = get_dataloaders(
    multimodal=(mode == "fusion"),
    data_dir=data_dir
)

# 3. Measure inference speed
infer_ms = measure_inference_time(model, test_loader, device)

# 4. Get all predictions
y_true, y_pred, logits_all = predict(model, test_loader, device, mode=args.mode)

# 5. Compute Top-5 accuracy
top5 = topk_accuracy(logits_all, torch.tensor(y_true), k=5)

# 6. Print metrics
metrics = compute_and_print_metrics(y_true, y_pred, classes, top5_acc=top5, infer_ms=infer_ms)

# 7. Save outputs
save_metrics(metrics, output_dir / f"{mode}_test_metrics.json")
plot_confusion_matrix(y_true, y_pred, classes, output_dir, prefix=mode)

# 8. Optionally: Grad-CAM
if args.gradcam:
    visualize_gradcam(...)
```

---

## Function: `predict(model, loader, device, mode)` → `(y_true, y_pred, logits)`

```python
@torch.no_grad()
def predict(model, loader, device, mode="fusion"):
    for imgs, tex, col, labels in loader:
        if mode == "fusion":
            tex, col = tex.to(device), col.to(device)
            logits, _ = model.forward_fusion(imgs, tex, col)
        else:
            logits = model.forward_image(imgs)

        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds, torch.cat(all_logits, dim=0)
```

Returns raw logits (not just predictions) so `topk_accuracy` can compute Top-5.

---

## Grad-CAM Sample Selection

```python
wrong  = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred != l]
right  = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred == l]
samples = wrong[:8] + right[:8]
```

Selects 8 misclassified + 8 correctly classified samples.
Showing errors helps diagnose failure modes (e.g., hard-negative confusion).
Showing correct cases confirms the model attends to the spice, not background.

---

## Output Files

All files written to `outputs/`:

| File | Contents |
|---|---|
| `{mode}_test_metrics.json` | Top-1, Top-5, F1, inference time |
| `{mode}_confusion_matrix.png` | Count + normalized confusion matrices |
| `{mode}_gradcam.png` | Grad-CAM overlays for 16 samples (if --gradcam) |

The `mode` prefix distinguishes fusion vs image-only runs:
- `fusion_test_metrics.json` — Phase 3 results
- `image_test_metrics.json` — Phase 1 results

---

## Strict Test Set Protocol

The test DataLoader uses `get_dataloaders()` with the same `RANDOM_SEED=42` split
used during training. This guarantees no data leakage — the exact same images
that were excluded from training appear in this evaluation.
