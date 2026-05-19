# batch_test.py — Bulk Inference Validator

## Overview

Runs inference on an entire folder of images and reports per-class and overall accuracy.
Designed as a robustness test — verifies the pipeline handles arbitrary real-world inputs
(any size, format, EXIF rotation, etc.) without crashing.

---

## Usage

```bash
python build_test_folder.py          # first: populate test/ folder
python batch_test.py                 # then: run batch inference on test/
python batch_test.py --folder test --mode fusion
python batch_test.py --folder /path/to/images --mode image --ckpt p1_best.pth
python batch_test.py --limit 10      # cap at 10 images per class
```

| Argument | Default | Description |
|---|---|---|
| `--folder` | `test/` | Root folder to scan |
| `--mode` | `"fusion"` | `"fusion"` or `"image"` |
| `--ckpt` | `best.pth` | Checkpoint file |
| `--limit` | `0` (all) | Max images per class (0 = unlimited) |

---

## Image Discovery: `collect_images(root)`

```python
def collect_images(root):
    cls_lookup = {c.lower(): c for c in config.CLASSES}   # case-insensitive

    for p in root.rglob("*"):
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        parent = p.parent.name.lower()

        if parent in cls_lookup:
            by_class[cls_lookup[parent]].append(p)   # labeled image
        else:
            unlabeled.append(p)                       # no class folder

    return by_class, unlabeled
```

**Folder structure expected:**
```
test/
├── turmeric/
│   ├── image1.jpg    ← labeled (true class = turmeric)
│   └── image2.png
├── cumin/
│   └── ...
└── some_unknown.jpg  ← unlabeled (no accuracy tracking)
```

Case-insensitive: `Turmeric/`, `TURMERIC/`, `turmeric/` all work.
Files with unknown parent folders go to `unlabeled` — inference runs but accuracy not tracked.

---

## Per-Image Inference: `predict_one(model, img_rgb, mode, device)`

```python
@torch.no_grad()
def predict_one(model, img_rgb, mode, device):
    x = preprocess_image(img_rgb).to(device)
    if mode == "fusion":
        tex_np, col_np = features.extract_all(img_rgb)
        tex = torch.from_numpy(tex_np).unsqueeze(0).to(device)
        col = torch.from_numpy(col_np).unsqueeze(0).to(device)
        logits, _ = model.forward_fusion(x, tex, col)
    else:
        logits = model.forward_image(x)

    probs = F.softmax(logits, dim=1)[0].cpu()
    return int(probs.argmax()), float(probs.max())
```

Returns `(predicted_class_index, confidence_score)`.
Confidence = softmax probability of top prediction.

---

## Main Loop and Error Handling

```python
for cls_name, paths in sorted(by_class.items()):
    true_idx = config.CLASSES.index(cls_name)
    for p in sample:
        try:
            img = load_image_rgb(str(p))
            pred_idx, conf = predict_one(model, img, args.mode, device)
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
            continue    # don't crash — record and move on
```

All exceptions are caught and recorded in `failed` list.
Batch continues even if individual images fail.

---

## Output Report

```
  ============================================================
  Per-class accuracy:
  class              correct    total    acc
  ----------------------------------------------
  black pepper           200      200  100.00%
  cardamom               198      200   99.00%
  coriander              194      200   97.00%
  cumin                  193      200   96.50%
  ...
  OVERALL                ...      ...   99.00%

  Throughput : 45.3 img/s  (48.7s total)

  Decode/inference failures: 0
```

**Exit codes:**
- `0` — all images processed without errors
- `2` — some images failed (non-zero `failed` list)

Useful for CI/CD integration — exit code 2 triggers failure.

---

## Unlabeled Images

```python
for p in unlabeled[:limit or None]:
    try:
        img = load_image_rgb(str(p))
        predict_one(model, img, args.mode, device)
        seen += 1
    except Exception as e:
        failed.append(...)
```

For unlabeled images, inference runs but no accuracy is tracked.
Purpose: confirm the pipeline doesn't crash on arbitrary input files.
