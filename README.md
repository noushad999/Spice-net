# SpiceFusionNet

**Deep Convolutional Neural Networks for Fine-Grained Spice Image Classification**

CSE 414 — Machine Learning and Deep Learning Lab
University of Asia Pacific

---

## Team

| Name | ID | Contribution |
|---|---|---|
| Md. Noushad Jahan Ramim | 22201257 | Model architecture, EfficientNet-B4, Phase 1 & 2 training |
| Maisha Sameha | 22201266 | Dataset preprocessing, augmentation pipeline, SAM variant |
| Samira Islam | 22201262 | Texture & color branches, AttentionFusion, ablation studies |
| Junaid Abedin Rafi | 22201265 | Baseline evaluation, Grad-CAM, results analysis |

**Supervisor:** Shahiar Raj, Lecturer, Dept. of CSE, University of Asia Pacific

---

## Results

| Model | Top-1 Acc | F1 Score | Inference |
|---|---|---|---|
| **SpiceFusionNet (ours)** | **99.68%** | **99.68%** | **2.70 ms/img** |
| ViT-Base/16 | 99.73% | 99.73% | 4.30 ms/img |
| EfficientNet-B4 (image only) | 99.59% | 99.59% | 2.10 ms/img |
| ResNet-50 | 99.36% | 99.36% | 1.85 ms/img |
| SVM (HOG + Color) | 32.49% | 30.69% | — |

---

## Architecture — SpiceFusionNet

Three-branch multi-modal fusion network:

```
Input Image (224×224×3)
        │
        ├── EfficientNet-B4 ──────────────────── 1792-d
        │   (ImageNet pretrained, Global Avg Pool)
        │
        ├── Texture Branch ──────────────────── 256-d
        │   LBP (10-d) + GLCM (48-d) = 58-d
        │   → MLP (58→128→256) + BN + ReLU
        │
        └── Color Branch ───────────────────── 128-d
            HSV Histogram (H:36+S:32+V:32) = 100-d
            → MLP (100→64→128) + BN + ReLU
                        │
                AttentionFusion
            (softmax-gated: a_img + a_tex + a_col = 1)
                        │
                   2176-d fused
                        │
                  fusion_head
              MLP(2176→512→11)
                        │
              11-class prediction
```

**Total parameters:** ~21.6M

---

## 3-Phase Training Strategy

| Phase | Epochs | Loss | What Trains |
|---|---|---|---|
| Phase 1 — Backbone Pretraining | 30 | CE + Label Smoothing | EfficientNet-B4 + img_head |
| Phase 2 — Contrastive Fine-tuning | 10 | SupCon Loss | Backbone + proj_head |
| Phase 3 — Full Fusion | 10 | 0.5×CE + 0.5×SupCon | All branches |

---

## Dataset — SpiceSpectrum

- **11,000 images** across 11 spice classes (~1,000 per class)
- **Class-balanced** — equal representation
- **Sources:** Open Images, iNaturalist, self-captured
- **Split:** 70% Train / 10% Val / 20% Test (seed=42, stratified)
- **11 Classes:** Black Pepper, Cardamom, Cinnamon, Cloves, Coriander, Cumin, Ginger, Nutmeg, Paprika, Saffron, Turmeric

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download SAM weights (optional, for ablation A5)
python download_sam.py

# Train — all 3 phases
python train.py

# Evaluate on test set
python evaluate.py --ckpt outputs/checkpoints/best.pth --mode fusion --gradcam

# Single image prediction
python predict.py path/to/spice.jpg

# Run baselines
python run_baselines.py

# Run ablation studies
python run_ablations.py --ablation all
```

---

## Project Structure

```
SpiceNet/
├── config.py                  # All hyperparameters
├── train.py                   # 3-phase training entry point
├── evaluate.py                # Test evaluation + Grad-CAM
├── predict.py                 # Single image inference
├── run_baselines.py           # B1–B4 baseline comparison
├── run_ablations.py           # A1–A5 ablation studies
├── run_svm.py                 # SVM classical baseline
├── sam_preprocess.py          # MobileSAM background removal
├── batch_test.py              # Bulk folder inference
├── build_test_folder.py       # Populate test/ from held-out split
├── test_sanity.py             # Shape/dimension unit tests
├── save_comparison.py         # Side-by-side model comparison
├── src/
│   ├── model.py               # SpiceFusionNet architecture
│   ├── dataset.py             # DataLoader, splits, augmentation
│   ├── features.py            # LBP + GLCM + HSV extraction
│   ├── losses.py              # SupConLoss + CombinedLoss
│   ├── trainer.py             # Phase 1/2/3 training loops
│   ├── baselines.py           # ResNet/ViT/EffNet/SVM
│   ├── gradcam.py             # Grad-CAM visualization
│   └── utils.py               # Metrics, plots, checkpoints
├── documentation/             # Full code documentation
├── spice_speech/              # Presentation speech scripts
└── outputs/                   # Training curves, confusion matrices
```

---

## Key Design Choices

- **Why EfficientNet-B4?** Best accuracy-speed tradeoff across 5 tested backbones (Ablation A4)
- **Why multi-modal?** Each branch captures what others miss — confirmed by Ablation A1
- **Why SupCon in Phase 2?** Critical for hard-negative pairs (Ablation A2) — 1%+ accuracy gain
- **Why spice-specific augmentation?** 93% → 99.68% improvement (Ablation A3)
- **Why not SAM background removal?** Grad-CAM shows model already focuses on spice (Ablation A5)

---

## References

1. M. Tan and Q. V. Le, "EfficientNet," ICML 2019
2. P. Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020
3. R. R. Selvaraju et al., "Grad-CAM," ICCV 2017
4. A. Kirillov et al., "Segment Anything," ICCV 2023
5. SpiceSpectrum Dataset, Data in Brief, Elsevier, 2025
