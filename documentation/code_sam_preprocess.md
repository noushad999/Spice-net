# sam_preprocess.py — SAM Background Removal Preprocessing

## Overview

Uses [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) (a lightweight version of
Meta's Segment Anything Model) to remove image backgrounds, saving clean spice-only images
to `Spice_Spectrum_SAM/`. Used in Ablation A5.

---

## Usage

```bash
python download_sam.py               # Download mobile_sam.pt (~38MB) first
python sam_preprocess.py             # Process all 11 classes
python sam_preprocess.py --cls turmeric    # Single class (for testing)
python sam_preprocess.py --model sam_b.pt  # Use larger SAM model
```

---

## How It Works

### Step 1: Center-Point Prompting

```python
def get_spice_mask(model, img_rgb):
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2   # image center

    results = model(img_rgb, points=[[cx, cy]], labels=[1], verbose=False)
```

SAM is a promptable segmentation model. Instead of segmenting everything,
we give it a "positive point" at the image center.

**Why center?**
In the SpiceSpectrum dataset, spice is almost always centered in the frame.
This simple heuristic works for the vast majority of images.

### Step 2: Best Mask Selection

```python
masks = results[0].masks.data.cpu().numpy()   # (N, H, W) — SAM may return multiple masks

# Score each mask by how much of it overlaps the center region
center_box = np.zeros((h, w), dtype=bool)
center_box[h//4:3*h//4, w//4:3*w//4] = True   # center 50% of image

scores = [np.logical_and(m.astype(bool), center_box).sum() for m in masks]
best = masks[np.argmax(scores)].astype(bool)
```

SAM returns multiple candidate masks. We pick the one with most overlap
with the center region (inner 50% of image).

### Step 3: Quality Filter

```python
ratio = best.sum() / (h * w)
if ratio < 0.05 or ratio > 0.95:
    return None   # mask failed quality check
```

Rejects masks that are too small (<5% of image — SAM missed the spice) or
too large (>95% — SAM grabbed the background instead of the spice).

### Step 4: Background Replacement

```python
def remove_background(img_rgb, mask, bg=(255, 255, 255)):
    out = img_rgb.copy()
    out[~mask] = np.array(bg)   # set non-spice pixels to white
    return out
```

Replaces background with white (255, 255, 255).
White was chosen because it's neutral and doesn't confuse HSV color features.

---

## Processing Loop

```python
def process_class(model, src_cls, dst_cls):
    for fp in tqdm(files):
        dst_fp = dst_cls / fp.name
        if dst_fp.exists():
            continue   # skip already processed (resume-safe)

        img = np.array(Image.open(fp).convert("RGB"))
        mask = get_spice_mask(model, img)

        if mask is not None:
            result = remove_background(img, mask)
        else:
            result = img     # fallback: copy original if SAM fails
            failed.append(str(fp))

        Image.fromarray(result).save(dst_fp, quality=95)
```

**Resume-safe**: skips already-processed files — safe to interrupt and restart.
**Graceful fallback**: if SAM fails for an image, copies the original unchanged.

---

## Failure Cases

SAM may fail when:
- Spice is not centered (e.g., multiple spices spread across frame)
- Very similar foreground/background colors
- Unusual lighting creates ambiguous edges

In these cases, the original image is copied (no background removal).
The `failed` list tracks these for review.

---

## Output

```
Spice_Spectrum_SAM/
├── black pepper/
│   ├── black_pepper_001.jpg   (white background, same filename)
│   └── ...
├── turmeric/
└── ...
```

Same directory structure and filenames as `Spice_Spectrum/`.
The dataset split (`build_splits()`) works identically with both directories.

---

## Performance

Processing ~11,000 images takes approximately:
- **GPU**: ~30–45 minutes
- **CPU**: ~3–6 hours

MobileSAM is ~60× faster than the full SAM model, making it practical for
preprocessing datasets of this size.
