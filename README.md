<div align="center">

<h1>🌶️ SpiceFusionNet</h1>

<p><strong>Deep Convolutional Neural Networks for Fine-Grained Spice Image Classification</strong></p>

<p>
  <img src="https://img.shields.io/badge/Accuracy-99.68%25-brightgreen?style=for-the-badge&logo=checkmarx"/>
  <img src="https://img.shields.io/badge/Top--5-99.95%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Inference-2.70ms%2Fimg-blue?style=for-the-badge&logo=lightning"/>
  <img src="https://img.shields.io/badge/Classes-11 Spices-orange?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/EfficientNet--B4-backbone-teal?style=flat-square"/>
  <img src="https://img.shields.io/badge/SupCon-contrastive-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-Research-lightgrey?style=flat-square"/>
</p>

<p>
  <b>CSE 414 — Machine Learning and Deep Learning Lab</b><br/>
  University of Asia Pacific &nbsp;|&nbsp; May 2026
</p>

</div>

---

## 👥 Team

<table align="center">
<tr>
  <th>Name</th>
  <th>ID</th>
  <th>Contribution</th>
</tr>
<tr>
  <td><b>Md. Noushad Jahan Ramim</b></td>
  <td><code>22201257</code></td>
  <td>Model architecture, EfficientNet-B4 backbone, Phase 1 & 2 training, report</td>
</tr>
<tr>
  <td><b>Maisha Sameha</b></td>
  <td><code>22201266</code></td>
  <td>Dataset preprocessing, augmentation pipeline, SAM variant creation, report</td>
</tr>
<tr>
  <td><b>Samira Islam</b></td>
  <td><code>22201262</code></td>
  <td>Texture & color branches, AttentionFusion module, ablation studies, report</td>
</tr>
<tr>
  <td><b>Junaid Abedin Rafi</b></td>
  <td><code>22201265</code></td>
  <td>Baseline evaluation, Grad-CAM visualization, results analysis, presentation</td>
</tr>
</table>

<p align="center"><b>Supervisor:</b> Shahiar Raj &nbsp;·&nbsp; Lecturer, Dept. of CSE, University of Asia Pacific</p>

---

## 🏆 Results at a Glance

<div align="center">

| Model | Top-1 Acc | F1 Score | Inference | Type |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **SpiceFusionNet (ours)** | **99.68%** | **99.68%** | **2.70 ms** | Multi-modal Fusion |
| 🥈 ViT-Base/16 | 99.73% | 99.73% | 4.30 ms | Vision Transformer |
| 🥉 EfficientNet-B4 (image only) | 99.59% | 99.59% | 2.10 ms | CNN (image only) |
| ResNet-50 | 99.36% | 99.36% | 1.85 ms | CNN |
| SVM (HOG + Color) | 32.49% | 30.69% | — | Classical ML |

</div>

> **Note:** SpiceFusionNet is **37% faster** than ViT-Base while achieving near-identical accuracy — making it the superior choice for real-world deployment.

---

## 🧠 Architecture

<div align="center">

```
                    ┌─────────────────────────────────────────┐
                    │         Input Image (224×224×3)         │
                    └──────────┬──────────┬───────────────────┘
                               │          │ (at 512×512, before aug)
                    ┌──────────▼──┐  ┌────▼──────────┐  ┌────────────────┐
                    │EfficientNet │  │ Texture Branch │  │  Color Branch  │
                    │    B4       │  │ LBP  + GLCM    │  │ HSV Histogram  │
                    │(pretrained) │  │ 10-d + 48-d    │  │ H:36+S:32+V:32 │
                    └──────┬──────┘  └───────┬────────┘  └───────┬────────┘
                           │                  │                    │
                        1792-d     MLP(58→128→256)      MLP(100→64→128)
                           │             256-d                  128-d
                           │                  │                    │
                    ┌──────▼──────────────────▼────────────────────▼──────┐
                    │                  AttentionFusion                     │
                    │     gate: Linear(2176→3) → Softmax                  │
                    │   α_img·f_cnn + α_tex·f_tex + α_col·f_col           │
                    └──────────────────────┬──────────────────────────────┘
                                           │ 2176-d
                                  ┌────────▼────────┐
                                  │   fusion_head   │
                                  │ MLP(2176→512→11)│
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  11-class output │
                                  │   Top-1: 99.68% │
                                  └─────────────────┘
```

</div>

<div align="center">

| Component | Parameters |
|:---|:---:|
| EfficientNet-B4 backbone | ~19.3 M |
| Texture + Color MLPs | ~56 K |
| AttentionFusion + fusion_head | ~2.2 M |
| **Total** | **~21.6 M** |

</div>

---

## 🔄 3-Phase Curriculum Training

<div align="center">

```
  Phase 1 (30 epochs)          Phase 2 (10 epochs)         Phase 3 (10 epochs)
  ─────────────────────        ─────────────────────        ─────────────────────
  Loss: CE + LabelSmooth       Loss: SupCon (τ=0.07)        Loss: 0.5×CE + 0.5×SupCon
  Trains: EfficientNet-B4      Trains: Backbone +           Trains: All branches +
          + img_head                   proj_head                     AttentionFusion
  Freezes: everything else     Freezes: all other parts     Freezes: nothing

  📦 → p1_best.pth             📦 → p2_last.pth             📦 → best.pth (final)
```

</div>

> **Why curriculum learning?** Training everything at once leads to local minima. Phase 2 Supervised Contrastive Learning is critical — it pushes hard-negative pairs (coriander ↔ cumin, paprika ↔ turmeric) apart **before** full fusion training begins.

---

## 📊 Dataset — SpiceSpectrum

<div align="center">

| Property | Value |
|:---|:---:|
| Total images | ~11,000 |
| Classes | 11 spice cultivars |
| Images per class | ~1,000 (class-balanced) |
| Train / Val / Test | 70% / 10% / 20% |
| Random seed | 42 (deterministic) |
| CNN input size | 224 × 224 |
| Feature extraction size | 512 × 512 |

</div>

**11 Spice Classes:**

```
Black Pepper  ·  Cardamom  ·  Cinnamon  ·  Cloves  ·  Coriander  ·  Cumin
Ginger  ·  Nutmeg  ·  Paprika  ·  Saffron  ·  Turmeric
```

**Hard-Negative Pairs** (visually confusable):

| Pair | Visual Similarity |
|:---|:---|
| Coriander ↔ Cumin | Small beige-brown seeds |
| Paprika ↔ Turmeric | Orange-yellow powder |
| Black Pepper ↔ Cloves | Dark spheroidal shape |
| Cinnamon ↔ Nutmeg | Brown powder / chips |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download SAM weights for background removal
python download_sam.py

# 3. Train — all 3 phases sequentially
python train.py

# 4. Evaluate on held-out test set
python evaluate.py --ckpt outputs/checkpoints/best.pth --mode fusion --gradcam

# 5. Predict a single image
python predict.py path/to/spice.jpg --topk 3

# 6. Run all baseline comparisons
python run_baselines.py

# 7. Run all 5 ablation studies
python run_ablations.py --ablation all
```

---

## 📂 Project Structure

<details>
<summary><b>Click to expand full project tree</b></summary>

```
SpiceNet/
│
├── config.py                    ← All hyperparameters (single source of truth)
├── train.py                     ← 3-phase training entry point
├── evaluate.py                  ← Test evaluation + confusion matrix + Grad-CAM
├── predict.py                   ← Single image inference (CLI)
├── run_baselines.py             ← B1–B4 baseline comparison
├── run_ablations.py             ← A1–A5 ablation studies
├── run_svm.py                   ← SVM classical baseline
├── sam_preprocess.py            ← MobileSAM background removal
├── batch_test.py                ← Bulk folder inference + accuracy report
├── build_test_folder.py         ← Populate test/ from held-out split
├── test_sanity.py               ← Shape/dimension unit tests
├── interactive_test.py          ← GUI demo
├── save_comparison.py           ← Side-by-side model comparison
│
├── src/
│   ├── model.py                 ← SpiceFusionNet + AttentionFusion
│   ├── dataset.py               ← DataLoader, 70/10/20 splits, augmentation
│   ├── features.py              ← LBP + GLCM + HSV feature extraction
│   ├── losses.py                ← SupConLoss + CombinedLoss
│   ├── trainer.py               ← Phase 1 / 2 / 3 training loops
│   ├── baselines.py             ← ResNet-50, ViT-Base, EfficientNet-B4, SVM
│   ├── gradcam.py               ← Grad-CAM hooks + heatmap visualization
│   └── utils.py                 ← Metrics, plots, checkpoint I/O
│
├── documentation/               ← 19 detailed code documentation files
│   ├── CODE_DOCS_INDEX.md
│   ├── code_model.md
│   ├── code_trainer.md
│   └── ...
│
├── spice_speech/                ← Presentation speech scripts (18 slides)
│   ├── MASTER_SPEECH_ALL_18_SLIDES.md
│   └── slide_01_title.md ... slide_18_references_thankyou.md
│
├── outputs/
│   ├── checkpoints/             ← Saved model weights (.pth)
│   ├── architecture_diagram.pptx
│   ├── *_confusion_matrix.png
│   ├── *_training_curves.png
│   └── *_gradcam.png
│
└── requirements.txt
```

</details>

---

## 🔬 Ablation Studies

<details>
<summary><b>A1 — Component Contribution</b></summary>

| Variant | Features | Top-1 |
|:---|:---|:---:|
| Image only | CNN 1792-d | ~97.5% |
| + Texture | CNN + LBP/GLCM | ~98.2% |
| + Color | CNN + HSV | ~98.0% |
| **Full Fusion** | **CNN + Texture + Color** | **99.68%** |

Every branch contributes. Full fusion is always best.
</details>

<details>
<summary><b>A2 — Contrastive Loss (Phase 2)</b></summary>

| Variant | Phase 2 | Top-1 |
|:---|:---:|:---:|
| Without SupCon | No | ~97.8% |
| **With SupCon** | **Yes** | **99.68%** |

Phase 2 is critical for hard-negative pairs like coriander ↔ cumin.
</details>

<details>
<summary><b>A3 — Augmentation Policy</b></summary>

| Variant | Top-1 |
|:---|:---:|
| No augmentation | ~93.0% |
| Standard (flip + crop) | ~96.5% |
| **Spice-specific (full)** | **99.68%** |

Domain-specific augmentation: +6.68% over no augmentation.
</details>

<details>
<summary><b>A4 — Backbone Selection</b></summary>

| Backbone | Params | Top-1 | Speed |
|:---|:---:|:---:|:---:|
| MobileNetV3-Large | 5.5M | ~93.5% | 0.8ms |
| EfficientNet-B0 | 5.3M | ~95.0% | 1.0ms |
| EfficientNet-B2 | 9.1M | ~96.8% | 1.4ms |
| ResNet-50 | 25.6M | ~94.2% | 1.85ms |
| **EfficientNet-B4** | **19.3M** | **99.68%** | **2.1ms** |

EfficientNet-B4 gives the best accuracy-to-latency tradeoff.
</details>

<details>
<summary><b>A5 — SAM Background Removal</b></summary>

| Variant | Data | Top-1 |
|:---|:---|:---:|
| Raw images | Spice_Spectrum/ | 99.68% |
| SAM cleaned | Spice_Spectrum_SAM/ | ~98.7% |

Background removal **does not help** — Grad-CAM confirms the model already focuses on the spice region naturally.
</details>

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tool |
|:---|:---|
| Language | Python 3.10 |
| Deep Learning | PyTorch · timm |
| Classical Features | scikit-image · OpenCV |
| Augmentation | Albumentations |
| Interpretability | Grad-CAM |
| Experiment Tracking | Weights & Biases |
| Model Sharing | HuggingFace Hub |

</div>

---

## 📚 References

```
[1] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019.
[2] P. Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020.
[3] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017.
[4] A. Kirillov et al., "Segment Anything," ICCV 2023.
[5] SpiceSpectrum Dataset, "Class-balanced dataset of commercially valuable spice cultivars,"
    Data in Brief, Elsevier, 2025.
[6] K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
```

---

<div align="center">

**Built with precision · Validated with ablations · Ready for deployment**

*University of Asia Pacific · CSE 414 · May 2026*

</div>
