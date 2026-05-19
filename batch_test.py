"""
End-to-end batch validator: run fusion inference on EVERY image in test/.

Proves the pipeline (load -> preprocess -> features -> model -> predict) is
robust to arbitrary inputs from disk — any extension, any mode, any size,
EXIF-rotated, etc. Reports per-class and overall accuracy.

Usage:
    python batch_test.py                       # uses test/ + best.pth fusion
    python batch_test.py --folder test --mode fusion
    python batch_test.py --folder /any/path --mode image --ckpt outputs/checkpoints/p1_best.pth
"""
import os, sys, time, argparse, traceback
from pathlib import Path
from collections import defaultdict

_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import numpy as np
import torch
import torch.nn.functional as F

import config
from src import features
from src.model import load_checkpoint
from interactive_test import load_image_rgb, preprocess_image, SUPPORTED_EXTS


def collect_images(root: Path):
    """Walk root/<class>/*.<ext>. Class is the immediate parent folder name.
    If the folder has no class subdirs, treat all images as unlabeled."""
    by_class = defaultdict(list)
    unlabeled = []

    # case-insensitive lookup of class names
    cls_lookup = {c.lower(): c for c in config.CLASSES}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        parent = p.parent.name.lower()
        if parent in cls_lookup:
            by_class[cls_lookup[parent]].append(p)
        else:
            unlabeled.append(p)
    return by_class, unlabeled


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(Path(_base) / "test"))
    ap.add_argument("--mode",   default="fusion", choices=["fusion", "image"])
    ap.add_argument("--ckpt",   default=None,
                    help="checkpoint (default: best.pth for fusion, p1_best.pth for image)")
    ap.add_argument("--limit",  type=int, default=0,
                    help="cap images per class (0 = all)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"  [ERROR] folder not found: {folder}")
        sys.exit(1)

    if args.ckpt is None:
        args.ckpt = str(config.CHECKPOINT_DIR /
                        ("best.pth" if args.mode == "fusion" else "p1_best.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    print(f"  Mode       : {args.mode}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Folder     : {folder}")

    by_class, unlabeled = collect_images(folder)
    total = sum(len(v) for v in by_class.values()) + len(unlabeled)
    print(f"  Images     : {total} ({len(by_class)} labeled classes, "
          f"{len(unlabeled)} unlabeled)")
    if total == 0:
        print("  [ERROR] no images found")
        sys.exit(1)

    print(f"\n  Loading model...")
    t0 = time.time()
    model, _, _, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    print(f"  Loaded in {(time.time()-t0):.1f}s")

    # ── Inference ─────────────────────────────────────────────────
    correct = 0
    seen = 0
    failed = []
    per_class = defaultdict(lambda: [0, 0])   # cls -> [correct, total]
    t0 = time.time()

    for cls_name, paths in sorted(by_class.items()):
        true_idx = config.CLASSES.index(cls_name)
        sample = paths if args.limit == 0 else paths[: args.limit]
        for p in sample:
            try:
                img = load_image_rgb(str(p))
                pred_idx, conf = predict_one(model, img, args.mode, device)
            except Exception as e:
                failed.append((p, f"{type(e).__name__}: {e}"))
                continue
            seen += 1
            ok = pred_idx == true_idx
            correct += ok
            per_class[cls_name][0] += ok
            per_class[cls_name][1] += 1

    # unlabeled: just confirm we can run inference without crashing
    for p in unlabeled[: args.limit or None]:
        try:
            img = load_image_rgb(str(p))
            predict_one(model, img, args.mode, device)
            seen += 1
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))

    dt = time.time() - t0

    # ── Report ────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  Per-class accuracy:")
    print(f"  {'class':<18s} {'correct':>10s} {'total':>8s} {'acc':>8s}")
    print(f"  {'-'*46}")
    for cls in config.CLASSES:
        c, t = per_class[cls]
        if t == 0:
            continue
        print(f"  {cls:<18s} {c:>10d} {t:>8d} {c/t*100:>7.2f}%")
    print(f"  {'-'*46}")
    if seen > 0:
        labeled_seen = sum(t for _, t in per_class.values())
        if labeled_seen > 0:
            print(f"  {'OVERALL':<18s} {correct:>10d} {labeled_seen:>8d} "
                  f"{correct/labeled_seen*100:>7.2f}%")
        print(f"\n  Throughput : {seen/dt:.1f} img/s  ({dt:.1f}s total)")

    print(f"\n  Decode/inference failures: {len(failed)}")
    for p, err in failed[:10]:
        print(f"    - {p.name}: {err}")
    if len(failed) > 10:
        print(f"    ... and {len(failed) - 10} more")

    sys.exit(0 if len(failed) == 0 else 2)


if __name__ == "__main__":
    main()
