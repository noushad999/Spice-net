# Inference Guide

## Evaluation on Test Split

`evaluate.py` runs the full held-out test evaluation and reports metrics.

```bash
# Evaluate Phase 3 fusion model (default)
python evaluate.py

# Specify checkpoint explicitly
python evaluate.py --ckpt outputs/checkpoints/best.pth --mode fusion

# Evaluate Phase 1 image-only model
python evaluate.py --ckpt outputs/checkpoints/p1_best.pth --mode image

# Generate Grad-CAM visualizations alongside evaluation
python evaluate.py --mode fusion --gradcam
```

### Output

| File | Description |
|---|---|
| `outputs/test_metrics.json` | Top-1, Top-5, F1 macro/weighted, per-class metrics, inference time |
| `outputs/confusion_matrix.png` | Normalized + raw count confusion matrix (2-subplot) |
| `outputs/fusion_gradcam.png` | Grad-CAM grid (8 misclassified + 8 random correct) |

---

## Batch Testing

`batch_test.py` runs inference on every image in a folder and reports per-class accuracy.
It handles EXIF rotation, unusual formats, and varying sizes robustly.

```bash
# Test against the test/ folder (default)
python batch_test.py

# Specify folder and model
python batch_test.py --folder test/ --mode fusion --ckpt outputs/checkpoints/best.pth

# Cap images per class (useful for quick sanity checks)
python batch_test.py --limit 20

# Image-only mode
python batch_test.py --mode image
```

### Output (console)

```
Processing black pepper:  100%|██████████| 200/200
Processing cardamom:      100%|██████████| 200/200
...

Per-class accuracy:
  black pepper   : 100.00%  (200/200)
  cardamom       :  99.50%  (199/200)
  ...
Overall accuracy : 99.00%
Throughput       : 370 img/s
```

---

## Interactive User Interface

`interactive_test.py` provides a menu-driven terminal interface for non-technical users.

```bash
python interactive_test.py
```

### Menu

```
==============================
       SpiceNet Identifier
==============================
1. Identify a spice from a photo
2. Pick a random photo from the dataset and identify
3. Identify every photo in a folder (batch mode)
a. Advanced options
q. Quit
```

### Usage: Option 1 (single photo)

Paste any file path when prompted. The tool resolves Windows paths (e.g., `C:\Users\...`),
handles EXIF rotation, and prints a friendly prediction:

```
Path: C:\Users\user\Desktop\mystery_spice.jpg

I'm very confident this is: turmeric  (98.7%)
Runner-up: paprika (0.9%)
```

### Usage: Option 3 (batch folder)

Enter a folder path. The tool processes every image and prints a per-image report:

```
black_pepper_001.jpg  →  black pepper   (99.2%)  [CORRECT]
black_pepper_002.jpg  →  black pepper   (97.8%)  [CORRECT]
cumin_019.jpg         →  coriander      (51.3%)  [WRONG — actual: cumin]
```

### Advanced Options (`a`)

- **Image-only mode:** Uses `p1_best.pth` instead of `best.pth` (no hand-crafted features)
- **Show feature vectors:** Prints the raw 58-d texture and 100-d color vectors

### Confidence Labels

The UI translates raw probability to plain English:

| Probability | Label |
|---|---|
| ≥ 95% | "I'm very confident this is:" |
| 80–95% | "I'm fairly confident this is:" |
| 60–80% | "I think this might be:" |
| < 60% | "I'm not sure, but my best guess is:" |

---

## Grad-CAM Visualization

Grad-CAM highlights which image regions the model focuses on when making predictions.
It hooks into EfficientNet-B4's last convolutional layer.

```bash
# Generate Grad-CAM as part of evaluate.py
python evaluate.py --gradcam

# Grad-CAM is also shown automatically in interactive_test.py (saved to outputs/)
```

### Output format

A 4×8 grid saved as `outputs/fusion_gradcam.png`:
- **Left column:** Original image
- **Right column:** Grad-CAM heatmap overlay
- **Green border:** Correctly classified
- **Red border:** Misclassified (with true label shown)

---

## Programmatic Inference (Python API)

```python
import torch
from PIL import Image
from src.model import SpiceFusionNet, load_checkpoint
from src.features import extract_all
from src.dataset import get_val_transform
import config

# Load model
model = SpiceFusionNet()
model, _ = load_checkpoint(model, "outputs/checkpoints/best.pth")
model.eval()

# Load image
img = Image.open("spice.jpg").convert("RGB")
transform = get_val_transform()
x = transform(img).unsqueeze(0)   # (1, 3, 224, 224)

# Extract hand-crafted features
import numpy as np
img_np = np.array(img.resize((512, 512)))
tex, col = extract_all(img_np)
tex_t = torch.tensor(tex, dtype=torch.float32).unsqueeze(0)  # (1, 58)
col_t = torch.tensor(col, dtype=torch.float32).unsqueeze(0)  # (1, 100)

# Inference
with torch.no_grad():
    logits, _ = model.forward_fusion(x, tex_t, col_t)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

print(f"Predicted: {config.CLASSES[pred]} ({probs[0, pred]:.1%})")
```

---

## Inference Speed

Measured on a single NVIDIA GPU, batch size 1 (worst case for latency):

| Model | Device | ms/image |
|---|---|---|
| SpiceFusionNet (fusion) | GPU | 2.70 |
| EfficientNet-B4 (image-only) | GPU | 2.10 |
| SpiceFusionNet (fusion) | CPU | ~45 |

Feature extraction (LBP + GLCM + HSV) adds ~3–5 ms on CPU per image when using full resolution.
For real-time applications, pre-extract features offline or use image-only mode.
