# Code Documentation Index

Complete explanation of every source file in the SpiceNet project.

---

## Core Library (`src/`)

| File | Documentation | What It Contains |
|---|---|---|
| `src/model.py` | [code_model.md](code_model.md) | SpiceFusionNet architecture, AttentionFusion, 3 forward modes, checkpoint I/O |
| `src/dataset.py` | [code_dataset.md](code_dataset.md) | Augmentation pipelines, train/val/test splits, SpiceDataset, DataLoaders |
| `src/features.py` | [code_features.md](code_features.md) | LBP (10-d), GLCM (48-d), HSV histogram (100-d) feature extraction |
| `src/losses.py` | [code_losses.md](code_losses.md) | SupConLoss (Phase 2), CombinedLoss 0.5×CE+0.5×SupCon (Phase 3) |
| `src/trainer.py` | [code_trainer.md](code_trainer.md) | Phase 1/2/3 training loops, early stopping, LR scheduling |
| `src/baselines.py` | [code_baselines.md](code_baselines.md) | ResNet-50, EfficientNet-B4, ViT-Base fine-tuning; SVM (HOG+Color) |
| `src/gradcam.py` | [code_gradcam.md](code_gradcam.md) | Grad-CAM hooks, heatmap generation, visualization grid |
| `src/utils.py` | [code_utils.md](code_utils.md) | set_seed, topk_accuracy, inference timing, plot curves/confusion matrix |

---

## Configuration

| File | Documentation | What It Contains |
|---|---|---|
| `config.py` | [code_config.md](code_config.md) | All hyperparameters, paths, class definitions, feature dimensions |

---

## Entry-Point Scripts

| File | Documentation | Purpose |
|---|---|---|
| `train.py` | [code_train.md](code_train.md) | Runs 3-phase training pipeline |
| `evaluate.py` | [code_evaluate.md](code_evaluate.md) | Test set evaluation, metrics, confusion matrix, Grad-CAM |
| `predict.py` | [code_predict.md](code_predict.md) | Single image inference with top-K output |
| `run_baselines.py` | [code_run_baselines.md](code_run_baselines.md) | Train + evaluate B1–B4 baseline models |
| `run_ablations.py` | [code_run_ablations.md](code_run_ablations.md) | A1–A5 ablation studies |

---

## Utility Scripts

| File | Documentation | Purpose |
|---|---|---|
| `sam_preprocess.py` | [code_sam_preprocess.md](code_sam_preprocess.md) | MobileSAM background removal → `Spice_Spectrum_SAM/` |
| `batch_test.py` | [code_batch_test.md](code_batch_test.md) | Bulk inference on a folder, per-class accuracy report |
| `build_test_folder.py` | [code_build_test_folder.md](code_build_test_folder.md) | Populate `test/` with held-out test images |
| `test_sanity.py` | [code_test_sanity.md](code_test_sanity.md) | Shape/dimension unit tests for features and model |

---

## Quick Reference: Data Flow

```
config.py
    ↓ paths + hyperparameters
src/dataset.py → build_splits() → 70/10/20 split
    ↓ image + hand-crafted features per sample
src/features.py → LBP(10d) + GLCM(48d) → texture(58d)
                  HSV histogram → color(100d)
    ↓
src/model.py → SpiceFusionNet
    ├── Phase 1: image → backbone → img_head → logits (CE)
    ├── Phase 2: image → backbone → proj_head → L2-norm (SupCon)
    └── Phase 3: image+tex+col → backbone+branches → AttentionFusion → fusion_head (CE+SupCon)
    ↓
src/trainer.py → PhaseTrainer.phase1/2/3
src/losses.py  → SupConLoss, CombinedLoss
src/utils.py   → metrics, plots, checkpoints
src/gradcam.py → heatmap visualization
```
