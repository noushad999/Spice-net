# src/gradcam.py — Gradient-weighted Class Activation Maps

## Overview

Grad-CAM produces heatmaps showing **which spatial regions of an image the model
focused on** when making a prediction. Red/hot areas = high attention.

For spice classification, this is useful to verify the model looks at the spice
itself (not the background, container, or label).

---

## How Grad-CAM Works (Concept)

1. Register hooks on a target convolutional layer to capture:
   - **Forward activations** — what features the layer computed
   - **Backward gradients** — how important each feature map is for the prediction

2. Run a forward pass → get prediction logits

3. Backpropagate gradients for the predicted class

4. Weight each feature map by its average gradient → sum across channels → ReLU

5. Resize the resulting heatmap to match input image size

6. Overlay on the original image

---

## Class: `GradCAM`

```python
class GradCAM:
    def __init__(self, model, target_layer=None, mode="fusion"):
```

### Target Layer Selection

```python
if target_layer is None:
    target_layer = model.backbone.conv_head
```

`conv_head` is the final 1×1 convolution in EfficientNet-B4 before global average pooling.
It is the last layer that still has spatial dimensions (H×W feature maps).
After this layer, global pooling collapses spatial information.

Why not `blocks[-1]`? `conv_head` is slightly earlier, giving a better spatial resolution
for the heatmap while still representing high-level semantic features.

### Hook Registration

```python
self._fwd = target_layer.register_forward_hook(self._save_activation)
self._bwd = target_layer.register_full_backward_hook(self._save_gradient)
```

PyTorch hooks fire automatically during forward/backward passes.
They store intermediate values in `self.activations` and `self.gradients`.

**Important:** Call `gcam.remove()` when done to avoid memory leaks from hooks.

### `__call__(x, tex, col, class_idx)` → `(cam, pred_idx)`

```python
def __call__(self, x, tex=None, col=None, class_idx=None):
    # Forward pass
    if mode == "fusion" and tex is not None:
        logits, _ = model.forward_fusion(x, tex, col)
    else:
        logits = model.forward_image(x)

    if class_idx is None:
        class_idx = logits.argmax(1).item()   # use predicted class

    # Backward pass for target class
    logits[0, class_idx].backward()

    # Compute CAM
    weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (C, 1, 1) global avg grad
    cam = (weights * self.activations).sum(dim=1)             # weighted sum over channels
    cam = F.relu(cam).squeeze().cpu().numpy()                 # ReLU removes negative
    cam = (cam - cam.min()) / (cam.max() + 1e-8)             # normalize to [0, 1]
    return cam, class_idx
```

**Step-by-step:**
1. `gradients`: shape `(B, C, H, W)` — gradients from backward pass
2. `mean(dim=(2,3))`: global average over spatial dims → `(B, C, 1, 1)` per-channel importance weights
3. Multiply weights × activations → weighted feature maps
4. Sum over channels → single heatmap `(H, W)`
5. ReLU: only keep positive activations (those that increase the class score)
6. Normalize to `[0, 1]` for visualization

---

## Helper: `_model_view(img_path)` → `(tensor, display, tex, col)`

This function solves a critical alignment problem: the heatmap must be overlaid
on the **exact same pixels** the model processed, not the original image.

```python
def _model_view(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))

    # Step 1: Resize+crop to get the display image (uint8, 224×224)
    crop_tf = A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224)])
    display = crop_tf(image=img)["image"]

    # Step 2: Normalize for the model (same pixels, just scaled)
    norm_tf = A.Compose([A.Normalize(...), ToTensorV2()])
    tensor = norm_tf(image=display)["image"].unsqueeze(0)

    # Step 3: Hand-crafted features from ORIGINAL image
    tex_np, col_np = extract_all(img)
    tex = torch.from_numpy(tex_np).unsqueeze(0).float()
    col = torch.from_numpy(col_np).unsqueeze(0).float()

    return tensor, display, tex, col
```

**Why is this important?**
If you show the original 512×512 image but the model processed a 224×224 crop,
the heatmap pixels won't align with the image pixels. This function ensures
`display` and `tensor` are derived from the exact same spatial crop.

Features are extracted from the original image (matching `dataset.py` behavior).

---

## Function: `visualize_gradcam()`

```python
def visualize_gradcam(model, img_paths, true_labels, classes,
                      device, output_path, n_samples=16, mode="fusion"):
```

Plots a grid of `n_samples` images with their Grad-CAM overlays.

### Layout
- `cols=4` → 4 image pairs per row (original + heatmap)
- Green title = correct prediction, Red title = wrong prediction

### Heatmap Overlay
```python
heatmap = cv2.resize(cam, (224, 224))                          # upscale from feature size
heatmap = cv2.applyColorMap(uint8(255*heatmap), COLORMAP_JET)  # false color
heatmap = cv2.cvtColor(heatmap, COLOR_BGR2RGB)                 # BGR→RGB for matplotlib
overlay = cv2.addWeighted(display, 0.5, heatmap, 0.5, 0)      # 50% blend
```

`COLORMAP_JET`: blue (low) → green (medium) → red (high attention).

### Mode Parameter

| mode | Forward pass used | Gradients from |
|---|---|---|
| `"fusion"` | `forward_fusion(x, tex, col)` | `fusion_head` logits |
| `"image"` | `forward_image(x)` | `img_head` logits |

Use `"fusion"` for the final model (Phase 3), `"image"` for Phase 1 comparison.

### Output
Saves to `outputs/{mode}_gradcam.png` at 150 DPI.

---

## Typical Usage (from evaluate.py)

```python
wrong = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred != l]
right = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred == l]
samples = wrong[:8] + right[:8]   # 8 errors + 8 correct predictions

visualize_gradcam(model, paths, labels, config.CLASSES, device,
                  output_path, n_samples=16, mode="fusion")
```

Showing misclassified examples helps diagnose failure modes.
