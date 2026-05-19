# src/features.py — Hand-Crafted Feature Extraction

## Overview

This module computes three types of traditional computer vision features from spice images.
These features capture texture and color information that CNN features sometimes miss for
fine-grained classification.

All functions take an RGB numpy array as input.

---

## Function: `extract_lbp(img_rgb)` → 10-d vector

**LBP = Local Binary Pattern**

### What it captures:
The micro-texture structure of a surface — whether it is rough, smooth, spotted, granular, etc.

### How it works:
1. Convert image to grayscale
2. For each pixel, compare its value to its `P=8` circular neighbors at radius `R=1`
3. Encode result as a binary number (1 if neighbor ≥ center, else 0)
4. Use `method="uniform"` — only count patterns with ≤ 2 bit-transitions (most common patterns)
5. Build a normalized histogram of these patterns

```python
def extract_lbp(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 8 + 2 = 10   # LBP_P + 2 uniform bins
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    return hist.astype(np.float32)   # shape: (10,)
```

### Output: 10-d normalized float32 vector

### Why LBP for spices?
- **Cumin vs Coriander**: both are small beige seeds, but cumin has fine ridges while
  coriander is smoother — LBP captures this difference
- **Whole vs Ground spices**: ground powder has uniform micro-texture; whole spices
  have structured surface patterns

---

## Function: `extract_glcm(img_rgb)` → 48-d vector

**GLCM = Gray-Level Co-occurrence Matrix**

### What it captures:
Repetitive spatial patterns and directionality — how often adjacent pixel pairs
have specific gray-level combinations.

### How it works:
1. Convert to grayscale, quantize to 64 levels (GLCM_LEVELS=64)
2. Build GLCM matrices for 2 distances × 4 angles = 8 combinations
3. Compute 6 statistical properties from each GLCM matrix

```python
def extract_glcm(img_rgb):
    gray = (gray // (256 // 64)).clip(0, 63)    # quantize to 64 levels

    glcm = graycomatrix(
        gray,
        distances=[1, 2],                        # short and medium range
        angles=[0°, 45°, 90°, 135°],            # 4 directions
        levels=64,
        symmetric=True, normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]
    # 6 props × 2 distances × 4 angles = 48 values
    return np.array([...]).astype(np.float32)    # shape: (48,)
```

### The 6 GLCM Properties:

| Property | What it measures |
|---|---|
| **Contrast** | Local intensity variation — high for rough textures |
| **Dissimilarity** | Similar to contrast but linear (not squared) |
| **Homogeneity** | How close GLCM values are to diagonal — high for smooth textures |
| **Energy** | Sum of squared GLCM entries — high for regular periodic patterns |
| **Correlation** | Linear dependency between gray values at offset pairs |
| **ASM** (Angular Second Moment) | Texture uniformity — high for uniform regions |

### Output: 48-d float32 vector (6 × 2 × 4)

---

## Function: `extract_texture(img_rgb)` → 58-d vector

```python
def extract_texture(img_rgb):
    return np.concatenate([extract_lbp(img_rgb), extract_glcm(img_rgb)])
    # 10 + 48 = 58
```

Simple concatenation of LBP and GLCM features.
This is the input to the texture branch MLP in `SpiceFusionNet`.

---

## Function: `extract_hsv(img_rgb)` → 100-d vector

**HSV = Hue, Saturation, Value color space**

### What it captures:
The color signature of the spice — turmeric's vivid yellow vs saffron's red-orange
vs black pepper's near-black.

### Why HSV instead of RGB?
- HSV separates **color** (H) from **brightness** (V) and **purity** (S)
- Robust to lighting changes: different V values under different lights, but H stays stable
- Matches how humans perceive color

```python
def extract_hsv(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # OpenCV HSV: H in [0, 180], S in [0, 256], V in [0, 256]
    h = np.histogram(hsv[:,:,0], bins=36, range=(0, 180), density=True)[0]  # 36-d
    s = np.histogram(hsv[:,:,1], bins=32, range=(0, 256), density=True)[0]  # 32-d
    v = np.histogram(hsv[:,:,2], bins=32, range=(0, 256), density=True)[0]  # 32-d

    return np.concatenate([h, s, v]).astype(np.float32)   # shape: (100,)
```

`density=True` normalizes each histogram to sum to 1.0 (probability distribution).
This makes the feature scale-invariant to image size.

### Output: 100-d float32 vector (H:36 + S:32 + V:32)

---

## Function: `extract_all(img_rgb)` → tuple(58-d, 100-d)

```python
def extract_all(img_rgb):
    return extract_texture(img_rgb), extract_hsv(img_rgb)
```

The single entry point called by `SpiceDataset.__getitem__()` and the inference scripts.
Returns both texture and color features in one call to minimize repeated grayscale conversions.

---

## Resolution Note

Features are extracted at the **original image resolution** (ideally 512×512).
Augmentation is applied AFTER feature extraction in the dataset pipeline.
This ensures LBP and GLCM capture true surface detail, not augmented artifacts.

See `config.py`:
```python
IMG_FULL = 512   # target resolution for feature extraction
IMG_SIZE = 224   # CNN input resolution
```

---

## scikit-image Dependency

LBP and GLCM use `scikit-image`:
```python
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
```

The module handles missing scikit-image gracefully:
```python
try:
    from skimage.feature import ...
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False
```

If not installed, `extract_lbp()` and `extract_glcm()` raise a clear error message
with the install command (`pip install scikit-image`).
