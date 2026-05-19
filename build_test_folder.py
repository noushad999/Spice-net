"""
Build D:/SpiceNet/test/ from the held-out test split (x_te).

These are images that the training pipeline has NEVER seen — guaranteed by the
deterministic seeded split (RANDOM_SEED=42) inside src/dataset.py:build_splits().

Usage:
    python build_test_folder.py                # 10 per class (default)
    python build_test_folder.py --per-class 20
    python build_test_folder.py --all          # copy every test-split image
    python build_test_folder.py --link         # symlink instead of copy (saves disk)
"""
import os, sys, shutil, argparse, random
from pathlib import Path
from collections import defaultdict

_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import config
from src.dataset import build_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10,
                    help="how many test images per class (default: 10)")
    ap.add_argument("--all", action="store_true",
                    help="copy ALL test-split images (overrides --per-class)")
    ap.add_argument("--link", action="store_true",
                    help="create symlinks instead of copying files")
    ap.add_argument("--out", type=str, default=str(Path(_base) / "test"),
                    help="output folder (default: <repo>/test)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Building held-out split (seed={config.RANDOM_SEED}, "
          f"ratios train/val/test = "
          f"{config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO})")
    _, _, _, _, x_te, y_te = build_splits()

    # Group test paths by class
    by_class = defaultdict(list)
    for p, lbl in zip(x_te, y_te):
        by_class[lbl].append(p)

    rng = random.Random(config.RANDOM_SEED)

    total_copied = 0
    print(f"\n  Class                Available   Selected")
    print(f"  " + "-" * 45)
    for idx, cls in enumerate(config.CLASSES):
        paths = by_class[idx]
        rng.shuffle(paths)
        selected = paths if args.all else paths[: args.per_class]

        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src in selected:
            src_p = Path(src)
            dst_p = cls_dir / src_p.name
            if dst_p.exists():
                continue
            if args.link:
                try:
                    os.symlink(src_p, dst_p)
                except OSError:
                    shutil.copy2(src_p, dst_p)
            else:
                shutil.copy2(src_p, dst_p)
            total_copied += 1

        print(f"  {cls:<18s}   {len(paths):>5d}    {len(selected):>5d}")

    print(f"\n  [OK] {total_copied} new file(s) written to {out_dir}")
    print(f"  [OK] These images were NOT used during training (split seed {config.RANDOM_SEED}).")
    print(f"\n  Tip: feed them into the interactive tester:")
    print(f"       python interactive_test.py")


if __name__ == "__main__":
    main()
