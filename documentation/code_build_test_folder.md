# build_test_folder.py — Test Set Population

## Overview

Copies held-out test images from `Spice_Spectrum/` into a `test/` folder
for use with `interactive_test.py` and `batch_test.py`.

The key guarantee: **every image in `test/` was excluded from training**.
The same seed and split ratios from `config.py` ensure deterministic holdout.

---

## Usage

```bash
python build_test_folder.py                  # 10 images per class (default)
python build_test_folder.py --per-class 50   # 50 per class
python build_test_folder.py --all            # copy all ~2200 test images
python build_test_folder.py --link           # symlinks instead of copies (saves disk)
python build_test_folder.py --out custom_dir/  # custom output location
```

| Argument | Default | Description |
|---|---|---|
| `--per-class` | `10` | Images to copy per class |
| `--all` | `False` | Copy all test-split images |
| `--link` | `False` | Create symlinks (saves disk space) |
| `--out` | `test/` | Output directory |

---

## How It Works

```python
# 1. Get the exact same test split used during training
_, _, _, _, x_te, y_te = build_splits()   # uses RANDOM_SEED=42

# 2. Group by class
by_class = defaultdict(list)
for p, lbl in zip(x_te, y_te):
    by_class[lbl].append(p)

# 3. Select and copy
rng = random.Random(config.RANDOM_SEED)   # deterministic shuffle
for idx, cls in enumerate(config.CLASSES):
    paths = by_class[idx]
    rng.shuffle(paths)
    selected = paths if args.all else paths[:args.per_class]

    for src in selected:
        dst = out_dir / cls / Path(src).name
        if args.link:
            os.symlink(src, dst)    # symlink (faster, saves space)
        else:
            shutil.copy2(src, dst)  # hard copy (safer for portability)
```

---

## Output Structure

```
test/
├── black pepper/
│   ├── black_pepper_007.jpg
│   └── ...   (10 images by default)
├── cardamom/
├── ...
└── turmeric/
```

Mirrors the `Spice_Spectrum/` structure — class name = folder name.
`batch_test.py` and `interactive_test.py` read this structure.

---

## Why This Script Exists

Instead of giving direct access to the full `Spice_Spectrum/` folder
(which contains training images), this script ensures you only see test images.

**Data leakage prevention:** If you test on training images, accuracy looks
higher than it actually is. This script enforces the evaluation protocol.

---

## Resume Safety

```python
if dst_p.exists():
    continue   # skip already-copied files
```

Safe to run multiple times — only copies new files.

---

## Printed Summary

```
  Class              Available   Selected
  -----------------------------------------
  black pepper          2,203         10
  cardamom              2,198         10
  coriander             2,205         10
  ...

  [OK] 110 new file(s) written to D:\SpiceNet\test
  [OK] These images were NOT used during training (split seed 42).
```
