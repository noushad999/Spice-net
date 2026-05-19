# run_baselines.py — Baseline Comparison Script

## Overview

Trains and evaluates all four baseline models and produces a comparison table
matching the paper's Table 1 (baseline results section).

---

## Usage

```bash
python run_baselines.py             # 20 epochs each (default)
python run_baselines.py --epochs 30  # more epochs for better convergence
```

---

## Execution Flow

```python
# 1. Shared DataLoaders (multimodal=False — baselines are image-only)
train_loader, val_loader, test_loader, _, _ = get_dataloaders(multimodal=False)
x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(config.DATA_DIR)

# 2. Run each neural baseline
for (name, model_fn) in [("resnet50", make_resnet50), ...]:
    model = model_fn()
    # fine-tune → load best → evaluate on test
    metrics, y_true, y_pred = run_nn_baseline(...)
    plot_confusion_matrix(y_true, y_pred, ...)

# 3. SVM baseline
val_acc, svm_pipe = train_svm(x_tr, y_tr, x_val, y_val)
X_te = build_svm_features(x_te)
y_pred_svm = svm_pipe.predict(X_te)
svm_metrics = compute_and_print_metrics(...)

# 4. Print summary table
# 5. Save to baseline_comparison.json
```

---

## Helper: `run_nn_baseline(name, model_fn, ...)`

```python
def run_nn_baseline(name, model_fn, train_loader, val_loader, test_loader, device, epochs, ckpt_dir):
    model = model_fn()                                  # create fresh model
    best_acc = finetune(model, ...)                     # fine-tune, saves best checkpoint
    state = torch.load(f"{name}_best.pth")              # reload best weights
    model.load_state_dict(state)
    infer_ms  = measure_inference_time(model, test_loader, device)
    y_true, y_pred = predict_nn(model, test_loader, device)
    metrics = compute_and_print_metrics(y_true, y_pred, classes, infer_ms=infer_ms)
    return metrics, y_true, y_pred
```

Pattern for each neural baseline:
1. Train with `finetune()` → saves `{name}_best.pth`
2. Reload best weights (not necessarily final epoch)
3. Measure inference speed
4. Get all test predictions
5. Compute metrics

---

## Summary Table Output

```
=================================================================
  Model                  Top-1 Acc   Macro F1  Infer(ms)
=================================================================
  resnet50                   0.9420     0.9418      1.85
  efficientnet_b4            0.9750     0.9748      2.10
  vit_base                   0.9610     0.9607      4.30
  svm_hog_color              0.8230     0.8215      N/A
=================================================================
```

---

## Checkpoint Outputs

| Baseline | Checkpoint | Description |
|---|---|---|
| B1 | `resnet50_best.pth` | Best ResNet-50 weights |
| B2 | `efficientnet_b4_best.pth` | Best EfficientNet-B4 (image-only) |
| B3 | `vit_base_best.pth` | Best ViT-Base weights |
| B4 | `svm_best.pkl` | Trained sklearn SVM pipeline |

## JSON Output

`outputs/baseline_comparison.json`:
```json
{
  "baselines": [
    {"model": "resnet50", "top1_accuracy": 0.942, "f1_macro": 0.9418, "inference_ms": 1.85},
    ...
  ]
}
```
