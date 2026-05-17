# Dataset Documentation

## SpiceSpectrum Dataset

| Property | Value |
|---|---|
| Total images | ~11,000 |
| Classes | 11 spice cultivars |
| Images per class | ~1,000 (class-balanced) |
| Image format | JPEG / PNG |
| Resolution | Variable (normalized to 224×224 for CNN, 512×512 for features) |
| License | Research use |

---

## Class List

| Index | Class Name | Notes |
|---|---|---|
| 0 | black pepper | Dark spheroidal peppercorns |
| 1 | cardamom | Green or brown pod |
| 2 | cinnamon | Brown rolled bark / powder |
| 3 | cloves | Dark nail-shaped bud |
| 4 | coriander | Small beige-brown seed |
| 5 | cumin | Similar to coriander — hard-negative pair |
| 6 | ginger | Gnarled pale-yellow rhizome |
| 7 | nutmeg | Brown seed / grated powder |
| 8 | paprika | Orange-red powder |
| 9 | saffron | Red-orange thread stigmas |
| 10 | turmeric | Bright yellow powder / rhizome |

---

## Directory Structure

```
Spice_Spectrum/
├── black pepper/
│   ├── black_pepper_001.jpg
│   ├── black_pepper_002.jpg
│   └── ...   (~1000 images)
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

---

## Train / Val / Test Splits

Splits are generated deterministically with `sklearn.train_test_split` and `RANDOM_SEED=42`.

| Split | Fraction | ~Images |
|---|---|---|
| Train | 70% | 7,700 |
| Validation | 10% | 1,100 |
| Test | 20% | 2,200 |

The same seed is used everywhere in the codebase, so the split is always identical.

**Important:** The test set is strictly held out. It is never used during training or hyperparameter
search. Use `build_test_folder.py` to copy test images for manual inspection without risking leakage.

---

## Building the Test Folder

`build_test_folder.py` copies held-out test images to `test/` for use with `interactive_test.py`
and `batch_test.py`.

```bash
# Copy 10 images per class (default)
python build_test_folder.py

# Copy all test-split images
python build_test_folder.py --all

# Custom number per class
python build_test_folder.py --per-class 50

# Use symlinks instead of copying (saves disk space)
python build_test_folder.py --link

# Custom output directory
python build_test_folder.py --out my_test_folder/
```

Result structure:
```
test/
├── black pepper/
├── cardamom/
...
└── turmeric/
```

---

## SAM-Preprocessed Variant (Spice_Spectrum_SAM)

`sam_preprocess.py` applies **MobileSAM** background removal to create a clean variant of the dataset.
The spice foreground is segmented using a center-point prompt; the background is replaced with white.

### When to use

Used in **Ablation A5** to measure the impact of background removal on classification accuracy.

### Running preprocessing

```bash
# Process all classes
python sam_preprocess.py

# Process a single class
python sam_preprocess.py --cls turmeric

# Use a different SAM model
python sam_preprocess.py --model sam_b.pt
```

Download SAM weights first if needed:
```bash
python download_sam.py
```

This downloads `mobile_sam.pt` (~38 MB) to the project root.

### Output structure

```
Spice_Spectrum_SAM/
├── black pepper/    # Same filenames as Spice_Spectrum, background removed
├── cardamom/
...
└── turmeric/
```

### Training with SAM data

```bash
python train.py --data_dir Spice_Spectrum_SAM
```

---

## Feature Extraction Details

Hand-crafted features are extracted in `src/features.py` at 512×512 resolution (before augmentation).

### LBP — Local Binary Pattern (10-d)

- **Method:** `uniform` (scikit-image)
- **Radius:** 3 pixels, 24 neighbor points
- **Output:** Normalized histogram over uniform patterns
- **Captures:** Fine-grained surface texture (rough vs smooth)

### GLCM — Gray-Level Co-occurrence Matrix (48-d)

- **Distances:** [1, 3] pixels
- **Angles:** [0°, 45°, 90°, 135°]
- **Properties:** contrast, dissimilarity, homogeneity, energy, correlation, ASM
- **Computation:** 6 properties × 2 distances × 4 angles = 48 values
- **Captures:** Repetitive texture patterns, directionality

### HSV Histogram (100-d)

- **Bins:** H: 36 bins, S: 32 bins, V: 32 bins
- **Normalization:** L1-normalized per channel
- **Captures:** Color signature (e.g., turmeric yellow vs saffron red)

### Combined Feature Vector

```
texture = [LBP (10-d)] + [GLCM (48-d)]  →  58-d
color   = [HSV H (36-d) + S (32-d) + V (32-d)]  →  100-d
```

Both vectors are computed per-image and stored alongside the image tensor in the dataset.
