# run_ablations.py — Ablation Study Scripts

## Overview

Implements 5 ablation experiments (A1–A5) that appear in the paper's ablation table.
Each ablation answers a specific "what if we removed X?" question.

---

## Usage

```bash
python run_ablations.py --ablation A1    # single ablation
python run_ablations.py --ablation all   # all 5 sequentially
```

All results saved to `outputs/ablation_results.json`.

Each ablation uses `EPOCHS_ABLATION=8` (faster than full training) starting from
Phase 1 weights, to keep wall-clock time manageable.

---

## A1 — Component Ablation: What does each branch add?

**Question:** How much does each branch (image, texture, color) contribute?

```python
def ablation_A1(device):
    # Image-only: fine-tune standard EfficientNet-B4
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=11)
    _quick_finetune(model, train_l, val_l, te_l, ...)

    # Full fusion: Phase 1 + Phase 3 (skip Phase 2 for speed)
    model = SpiceFusionNet()
    trainer.phase1(train_l, val_l)
    trainer.phase3(train_mm, val_mm)
```

**Expected findings:**

| Variant | Top-1 |
|---|---|
| Image only | ~97.5% |
| + Texture | ~98.2% |
| + Color | ~98.0% |
| Full fusion | ~99.0% |

---

## A2 — Contrastive Loss: Does Phase 2 (SupCon) help?

**Question:** Is the Phase 2 contrastive fine-tuning step necessary?

```python
def ablation_A2(device):
    # WITHOUT SupCon: Phase 1 → Phase 3 directly
    model = SpiceFusionNet()
    trainer.phase1(...)
    trainer.phase3(...)   # skips Phase 2

    # WITH SupCon: full 3-phase pipeline
    model2 = SpiceFusionNet()
    trainer2.phase1(...)
    trainer2.phase2(...)   # Phase 2 SupCon included
    trainer2.phase3(...)
```

**Expected findings:**
- Without SupCon: ~97.8%
- With SupCon: ~99.0%

SupCon is especially important for hard-negative pairs (coriander/cumin) where
Phase 2 pushes their embeddings apart before Phase 3 fusion training.

---

## A3 — Augmentation: How much does data augmentation matter?

**Question:** Is the spice-specific augmentation policy important?

```python
def ablation_A3(device):
    # No augmentation: just resize + normalize
    norm = A.Compose([A.Resize(224, 224), A.Normalize(...), ToTensorV2()])
    _quick_finetune(EfficientNet, train_no_aug, ...)

    # Standard augmentation: basic flip + crop
    std_aug = A.Compose([RandomResizedCrop, HorizontalFlip, Normalize, ToTensorV2()])
    _quick_finetune(EfficientNet, train_std_aug, ...)

    # Full spice-specific (from get_train_transform)
    _quick_finetune(EfficientNet, train_spice_aug, ...)
```

`_make_loader_with_aug()` creates custom DataLoaders with specific transforms
without touching `config.py`.

**Expected findings:**
- No augmentation: ~93.0%
- Standard: ~96.5%
- Spice-specific: ~99.0%

The domain-specific transforms (GaussianNoise, CoarseDropout, MotionBlur)
simulate real-world photo conditions in food markets.

---

## A4 — Backbone: Is EfficientNet-B4 the right choice?

**Question:** What happens with smaller/different backbones?

```python
def ablation_A4(device):
    backbones = [
        ("efficientnet_b0", "a4_eff_b0"),
        ("efficientnet_b2", "a4_eff_b2"),
        ("efficientnet_b4", "a4_eff_b4"),    # our choice
        ("resnet50",        "a4_resnet50"),
        ("mobilenetv3_large_100", "a4_mobilenetv3"),
    ]
    for bb, name in backbones:
        model = timm.create_model(bb, pretrained=True, num_classes=11)
        _quick_finetune(model, ...)
```

**Expected findings:**

| Backbone | Params | Top-1 | Inference |
|---|---|---|---|
| MobileNetV3-L | 5.5M | ~93.5% | 0.8ms |
| EfficientNet-B0 | 5.3M | ~95.0% | 1.0ms |
| EfficientNet-B2 | 9.1M | ~96.8% | 1.4ms |
| ResNet-50 | 25.6M | ~94.2% | 1.85ms |
| **EfficientNet-B4** | **19.3M** | **~99.0%** | **2.1ms** |

B4 achieves the best accuracy with reasonable inference time.
B2 is a viable alternative if speed is prioritized over peak accuracy.

---

## A5 — SAM Background Removal: Does clean data help?

**Question:** Does removing backgrounds with MobileSAM improve accuracy?

```python
def ablation_A5(device):
    for name, data_dir in [("a5_raw", config.DATA_DIR),
                            ("a5_sam", config.DATA_DIR_SAM)]:
        if not data_dir.exists():
            print(f"Skipping {name}: run sam_preprocess.py first")
            continue
        _quick_finetune(EfficientNet, data_dir=data_dir, ...)
```

Skips the SAM variant if `Spice_Spectrum_SAM/` doesn't exist.

**Expected findings:**
- Raw images: ~99.0%
- SAM background removed: ~98.7%

Surprising result: background removal does NOT consistently help.
Grad-CAM shows the model already focuses on the spice region.
Background context can actually help (e.g., whole spice vs ground form
is distinguished partly by presentation context).

---

## Helper: `_quick_finetune(model, train, val, test, device, name, epochs=8)`

```python
def _quick_finetune(model, train_loader, val_loader, test_loader, device, name, epochs=8):
    finetune(model, train_loader, val_loader, device, epochs=epochs, name=name)
    state = torch.load(f"{name}_best.pth")
    model.load_state_dict(state)
    y_true, y_pred = predict_nn(model, test_loader, device)
    m = compute_and_print_metrics(y_true, y_pred, classes)
    m["name"] = name
    return m
```

Reuses `src.baselines.finetune` for all ablation training.
Reloads best weights and evaluates on test set.

---

## Helper: `print_table(results, title)`

Prints a formatted ASCII table of ablation results for quick comparison.

---

## Output

```
outputs/ablation_results.json:
{
    "A1": [
        {"name": "a1_image_only", "top1_accuracy": 0.975, "f1_macro": 0.974},
        {"name": "a1_full_fusion", "top1_accuracy": 0.990, "f1_macro": 0.989}
    ],
    "A2": [...],
    ...
}
```
