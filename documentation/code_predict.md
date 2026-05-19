# predict.py — Single Image Inference

## Overview

Command-line tool to classify a single spice image using the trained model.
Shows top-K predictions with confidence scores and a bar chart.

---

## Usage

```bash
python predict.py path/to/spice.jpg
python predict.py D:\SpiceNet\test\turmeric\001.jpg --topk 5
python predict.py image.jpg --ckpt outputs/checkpoints/p1_best.pth --topk 3
```

| Argument | Default | Description |
|---|---|---|
| `image` | (required) | Path to input image |
| `--ckpt` | `best.pth` | Checkpoint to use |
| `--topk` | `3` | Number of top predictions to display |

---

## Image Loading: `load_image_rgb(path)` → numpy array

```python
def load_image_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)   # fix rotation from EXIF metadata
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)
```

`ImageOps.exif_transpose`: phones store rotation in EXIF metadata.
Without this, a photo taken in portrait mode would appear rotated sideways.

---

## Preprocessing: `preprocess(img_rgb)` → tensor

```python
def preprocess(img_rgb):
    tfm = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return tfm(image=img_rgb)["image"].unsqueeze(0)   # (1, 3, 224, 224)
```

Matches `get_val_transform()` exactly — same resize→crop→normalize pipeline
used during training's validation phase.

---

## Inference Flow

```python
# 1. Load model
model, _, _, _ = load_checkpoint(args.ckpt, device)
model.eval()

# 2. Load and preprocess image
img = load_image_rgb(args.image)
x   = preprocess(img).to(device)

# 3. Extract hand-crafted features
tex_np, col_np = extract_all(img)
tex = torch.from_numpy(tex_np).unsqueeze(0).float().to(device)
col = torch.from_numpy(col_np).unsqueeze(0).float().to(device)

# 4. Fusion prediction
with torch.no_grad():
    logits, _ = model.forward_fusion(x, tex, col)
    probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()   # (11,) probabilities

# 5. Sort and display top-K
top_indices = probs.argsort()[::-1][:args.topk]
```

---

## Output Format

```
==================================================
  Prediction for: turmeric_001.jpg
==================================================
  1. turmeric        92.47%  ######################################### <<<
  2. paprika          5.12%  ##
  3. ginger           1.83%  
==================================================
  Final answer: TURMERIC
==================================================
```

The bar visualization (`#` characters) scales to 40 characters for 100%.
`<<<` marks the top prediction.

---

## Notes

- Always uses **fusion mode** (full model) for inference
- Runs on GPU if available, CPU otherwise
- EXIF-aware: handles rotated mobile phone photos
- `extract_all()` called on original image before preprocessing
