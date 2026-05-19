# SpiceNet — Documentation Overview

**SpiceNet** is a multi-modal deep learning framework for fine-grained spice image classification.
It achieves **99.00% Top-1 accuracy** across 11 commercially valuable spice cultivars by fusing
CNN features with hand-crafted texture and color descriptors.

---

## Documentation Index

| File | Contents |
|---|---|
| [architecture.md](architecture.md) | Model design, feature dimensions, forward modes |
| [training_guide.md](training_guide.md) | 3-phase training pipeline, CLI usage, checkpoints |
| [dataset.md](dataset.md) | SpiceSpectrum dataset, splits, augmentation, SAM preprocessing |
| [experiments.md](experiments.md) | Baselines (B1–B4), ablations (A1–A5), result tables |
| [api_reference.md](api_reference.md) | All modules, classes, and functions with signatures |
| [inference_guide.md](inference_guide.md) | Batch testing, interactive UI, Grad-CAM |

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision timm albumentations scikit-learn scikit-image opencv-python matplotlib seaborn
```

### 2. Prepare dataset
Place the `Spice_Spectrum/` directory in the project root. It must contain one subfolder per class:
```
Spice_Spectrum/
├── black pepper/
├── cardamom/
├── cinnamon/
├── cloves/
├── coriander/
├── cumin/
├── ginger/
├── nutmeg/
├── paprika/
├── saffron/
└── turmeric/
```

### 3. Run full training (all 3 phases)
```bash
python train.py
```

### 4. Evaluate on test split
```bash
python evaluate.py --ckpt outputs/checkpoints/best.pth --mode fusion --gradcam
```

### 5. Interactive identification
```bash
python interactive_test.py
```

---

## Key Results

| Model | Top-1 Acc | Top-5 Acc | Inference (ms/img) |
|---|---|---|---|
| SpiceFusionNet (ours) | **99.00%** | **99.95%** | 2.70 |
| EfficientNet-B4 (image-only) | 97.50% | 99.80% | 2.10 |
| ResNet-50 | 94.20% | 99.30% | 1.85 |
| ViT-Base | 96.10% | 99.60% | 4.30 |
| SVM (HOG + Color) | 82.30% | 97.10% | 0.15 |

---

## Hardware Requirements

- **Training:** NVIDIA GPU with ≥8 GB VRAM recommended (EfficientNet-B4 backbone)
- **Inference:** CPU-only supported; GPU reduces latency from ~15 ms to ~2.7 ms per image
- **Storage:** ~2 GB for dataset, ~500 MB for checkpoints, ~100 MB for SAM weights

---

## 11 Spice Classes

```
0  black pepper    1  cardamom    2  cinnamon
3  cloves          4  coriander   5  cumin
6  ginger          7  nutmeg      8  paprika
9  saffron         10 turmeric
```

Hard-negative pairs (visually confusable):

| Pair | Visual Similarity |
|---|---|
| coriander ↔ cumin | small beige seeds |
| paprika ↔ turmeric | orange-yellow powder |
| black pepper ↔ cloves | dark spheroidal shape |
| cinnamon ↔ nutmeg | brown powder / chips |
