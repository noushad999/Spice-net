"""
SpiceNet - AI spice identifier (interactive).

Designed for users with zero ML background. Pick a menu option, give a photo,
get a plain-English answer.

Usage:
    python interactive_test.py
"""
import os, sys, time, random, re
from pathlib import Path
from typing import Optional, List, Tuple

_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

import config
from src import features
from src.model import load_checkpoint


# ─── Checkpoints — fixed, no user choice ──────────────────────────────────────
CKPT_DIR     = config.CHECKPOINT_DIR
FUSION_CKPT  = CKPT_DIR / "best.pth"        # Phase-3 full fusion
IMAGE_CKPT   = CKPT_DIR / "p1_best.pth"     # Phase-1 image-only

# ─── In-process model cache (load once per session) ───────────────────────────
_CACHE = {}

def get_model(ckpt_path: Path, device):
    key = str(ckpt_path)
    if key not in _CACHE:
        print(f"  Loading AI model... ", end="", flush=True)
        t0 = time.time()
        model, _, _, _ = load_checkpoint(str(ckpt_path), device)
        model.eval()
        _CACHE[key] = model
        print(f"done ({time.time()-t0:.1f}s)")
    return _CACHE[key]


# ─── Image loading (robust to any input) ──────────────────────────────────────
def load_image_rgb(path: str) -> np.ndarray:
    """Open ANY image robustly: EXIF-rotated, any mode (RGBA/L/P/CMYK), tiny or huge."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in SUPPORTED_EXTS:
        print(f"  [warn] unusual extension '{p.suffix}' — attempting to decode anyway")

    try:
        with Image.open(p) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"could not decode image: {p} ({type(e).__name__}: {e})")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"unexpected image shape {arr.shape} from {p}")
    h, w = arr.shape[:2]
    if min(h, w) < 8:
        raise RuntimeError(f"image too small: {w}x{h} (need >=8 px)")
    return arr


def preprocess_image(img_rgb: np.ndarray) -> torch.Tensor:
    resize = int(config.IMG_SIZE * 256 / 224)
    tfm = A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])
    return tfm(image=img_rgb)["image"].unsqueeze(0)


# ─── Path resolution: handle Windows paths in WSL + fuzzy suggestions ─────────
def _resolve_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    s = raw.strip().strip('"').strip("'")

    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m and os.path.exists("/mnt"):
        s = f"/mnt/{m.group(1).lower()}/{m.group(2)}"

    s = s.replace("\\", "/")
    base = Path(_base)

    candidates = [Path(s).expanduser()]
    if not Path(s).is_absolute():
        candidates += [
            base / s,
            base / "test" / s,
            base / "Spice_Spectrum" / s,
        ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c
        if c.parent.exists() and c.parent.is_dir():
            target = c.name.lower()
            for sib in c.parent.iterdir():
                if sib.name.lower() == target and sib.is_file():
                    return sib
    return None


def _suggest_similar(raw: str, k: int = 3) -> List[Path]:
    """Find images whose basename contains a similar token (best-effort hint)."""
    test_dir = Path(_base) / "test"
    if not test_dir.exists():
        return []
    needle = Path(raw).stem.lower().replace(" ", "_")
    hits = []
    for p in test_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            if needle and needle in p.stem.lower():
                hits.append(p)
                if len(hits) >= k:
                    break
    return hits


# ─── Friendly prompts ─────────────────────────────────────────────────────────
def ask_image() -> Optional[str]:
    print()
    print("  Tip: paste any path (Windows or Linux style). Examples:")
    print("       test/turmeric/turmeric_19.jpg")
    print("       D:\\SpiceNet\\test\\saffron\\saffron_5.jpg")
    print("       (or just press Enter to go back)")
    while True:
        try:
            raw = input("\n  Photo path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        resolved = _resolve_path(raw)
        if resolved:
            return str(resolved)
        print(f"  -> could not find that file.")
        for s in _suggest_similar(raw):
            try:
                rel = s.relative_to(_base)
            except ValueError:
                rel = s
            print(f"     did you mean: {rel}?")


def ask_class() -> Optional[int]:
    print("\n  Pick a spice (we'll grab a random photo of it):")
    cols = 3
    for i, c in enumerate(config.CLASSES):
        end = "\n" if (i + 1) % cols == 0 else "    "
        print(f"   {i+1:>2}. {c:<14s}", end=end)
    if len(config.CLASSES) % cols:
        print()
    print(f"    0. Surprise me!")
    try:
        s = input("\n  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not s.isdigit():
        return None
    n = int(s)
    if n == 0:
        return random.randint(0, len(config.CLASSES) - 1)
    if 1 <= n <= len(config.CLASSES):
        return n - 1
    return None


def random_sample(cls_idx: int) -> Optional[Path]:
    cls_name = config.CLASSES[cls_idx]
    test_dir = Path(_base) / "test" / cls_name
    if not test_dir.exists():
        print(f"  [ERROR] No test photos for '{cls_name}'. "
              f"Run 'python build_test_folder.py' first.")
        return None
    candidates = [p for p in test_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not candidates:
        return None
    return random.choice(candidates)


# ─── Output formatting ────────────────────────────────────────────────────────
def _confidence_label(p: float) -> str:
    if p >= 0.90: return "I'm very confident"
    if p >= 0.70: return "I'm confident"
    if p >= 0.50: return "I think"
    if p >= 0.30: return "My best guess is"
    return "I'm not sure, but possibly"


def _show_result(probs: torch.Tensor, true_label: Optional[str] = None):
    top1 = int(probs.argmax())
    name = config.CLASSES[top1]
    conf = float(probs[top1])

    print()
    print(f"  {_confidence_label(conf)}: {name.upper()}  ({conf*100:.1f}%)")
    if true_label is not None:
        ok = (name == true_label)
        mark = "[CORRECT]" if ok else "[WRONG]"
        print(f"  {mark}  actual label: {true_label}")

    print(f"\n  My top 3 guesses:")
    p, idx = probs.topk(min(3, len(config.CLASSES)))
    for rank, (pp, ii) in enumerate(zip(p.tolist(), idx.tolist()), 1):
        bar = "#" * max(1, int(pp * 40))
        print(f"    {rank}. {config.CLASSES[ii]:<14s} {pp*100:5.1f}%  {bar}")


# ─── Inference paths ──────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_fusion(img_rgb, device) -> torch.Tensor:
    model = get_model(FUSION_CKPT, device)
    x = preprocess_image(img_rgb).to(device)
    tex_np, col_np = features.extract_all(img_rgb)
    tex = torch.from_numpy(tex_np).unsqueeze(0).to(device)
    col = torch.from_numpy(col_np).unsqueeze(0).to(device)
    logits, _ = model.forward_fusion(x, tex, col)
    return F.softmax(logits, dim=1)[0].cpu()


@torch.no_grad()
def _predict_image_only(img_rgb, device) -> torch.Tensor:
    model = get_model(IMAGE_CKPT, device)
    x = preprocess_image(img_rgb).to(device)
    logits = model.forward_image(x)
    return F.softmax(logits, dim=1)[0].cpu()


def _identify_one(img_path: str, device, true_label: Optional[str] = None):
    img = load_image_rgb(img_path)
    h, w = img.shape[:2]
    print(f"  Loaded photo ({w}x{h}).  Analyzing...")
    t0 = time.time()
    probs = _predict_fusion(img, device)
    print(f"  Done in {(time.time()-t0)*1000:.0f} ms.")
    _show_result(probs, true_label=true_label)


def _identify_folder(folder: Path, device):
    paths = []
    cls_lookup = {c.lower(): c for c in config.CLASSES}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
    if not paths:
        print(f"  [ERROR] No image files found in {folder}.")
        return

    print(f"  Found {len(paths)} photo(s). Loading model first...")
    get_model(FUSION_CKPT, device)
    print(f"  Analyzing...")

    correct = 0
    labeled = 0
    failed  = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        # Progress every 10 images
        if i % 10 == 0 or i == len(paths):
            print(f"    [{i}/{len(paths)}]", end="\r", flush=True)
        try:
            img = load_image_rgb(str(p))
            probs = _predict_fusion(img, device)
            pred = config.CLASSES[int(probs.argmax())]
        except Exception:
            failed += 1
            continue
        true = cls_lookup.get(p.parent.name.lower())
        if true is not None:
            labeled += 1
            if pred == true:
                correct += 1
    dt = time.time() - t0
    print()
    print(f"\n  Analyzed {len(paths) - failed} photo(s) in {dt:.1f}s "
          f"({(len(paths)-failed)/dt:.1f} per second).")
    if labeled:
        print(f"  Correct: {correct}/{labeled}  ({correct/labeled*100:.1f}%)")
    if failed:
        print(f"  Could not read: {failed} file(s)")


# ─── Advanced (hidden behind menu option 'a') ─────────────────────────────────
def _advanced_menu(device):
    print("\n  Advanced options:")
    print("    1. Image-only model prediction (no texture/color features)")
    print("    2. Show feature vectors (LBP / GLCM / HSV)")
    print("    b. Back")
    try:
        c = input("\n  Choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if c == "b" or not c:
        return

    img_path = ask_image()
    if not img_path:
        return
    img = load_image_rgb(img_path)
    h, w = img.shape[:2]
    print(f"  Loaded photo ({w}x{h}).")

    if c == "1":
        probs = _predict_image_only(img, device)
        _show_result(probs)
    elif c == "2":
        lbp  = features.extract_lbp(img)
        glcm = features.extract_glcm(img)
        hsv  = features.extract_hsv(img)
        print(f"\n  LBP   (texture pattern)   : {lbp.shape[0]} numbers   mean={lbp.mean():.4f}")
        print(f"  GLCM  (texture statistics): {glcm.shape[0]} numbers   mean={glcm.mean():.4f}")
        print(f"  HSV   (color histogram)   : {hsv.shape[0]} numbers   mean={hsv.mean():.4f}")
    else:
        print("  -> invalid; back to main menu.")


# ─── Main loop ────────────────────────────────────────────────────────────────
MENU = """
============================================================
  SpiceNet - AI Spice Identifier
============================================================
  I can recognize 11 spices: black pepper, cardamom, cinnamon,
  cloves, coriander, cumin, ginger, nutmeg, paprika, saffron,
  turmeric.

  What would you like to do?

   1. Identify a spice from a photo
   2. Pick a random photo from our dataset and identify it
   3. Identify every photo in a folder
   a. Advanced options
   q. Quit
============================================================"""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_label = "GPU" if device.type == "cuda" else "CPU"
    print(f"\n  Using {dev_label}.")

    if not FUSION_CKPT.exists():
        print(f"\n  [ERROR] Trained model not found at:\n          {FUSION_CKPT}")
        print(f"  Train one first with: python train.py")
        return

    while True:
        print(MENU)
        try:
            choice = input("\n  Choice [1]: ").strip().lower() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            return

        if choice == "q":
            print("  Bye.")
            return

        try:
            if choice == "1":
                p = ask_image()
                if p:
                    _identify_one(p, device)

            elif choice == "2":
                cls_idx = ask_class()
                if cls_idx is None:
                    continue
                p = random_sample(cls_idx)
                if not p:
                    continue
                try:
                    rel = p.relative_to(_base)
                except ValueError:
                    rel = p
                print(f"\n  Selected photo: {rel}")
                _identify_one(str(p), device, true_label=config.CLASSES[cls_idx])

            elif choice == "3":
                print()
                print("  Tip: enter a folder path. Default is the 'test' folder.")
                try:
                    raw = input("  Folder [test]: ").strip() or "test"
                except (EOFError, KeyboardInterrupt):
                    continue
                # Reuse resolver but require a directory
                cand = Path(raw)
                if not cand.is_absolute():
                    m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
                    if m and os.path.exists("/mnt"):
                        raw = f"/mnt/{m.group(1).lower()}/{m.group(2)}"
                    raw = raw.replace("\\", "/")
                    cand = Path(raw)
                    if not cand.is_absolute():
                        cand = Path(_base) / cand
                if not cand.exists() or not cand.is_dir():
                    print(f"  [ERROR] not a folder: {cand}")
                    continue
                _identify_folder(cand, device)

            elif choice == "a":
                _advanced_menu(device)

            else:
                print(f"  -> please pick 1, 2, 3, a, or q.")
                continue

        except FileNotFoundError as e:
            print(f"  [ERROR] file not found: {e}")
        except KeyboardInterrupt:
            print("\n  Cancelled.")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")

        try:
            again = input("\n  Try another? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            return
        if again == "n":
            print("  Bye.")
            return


if __name__ == "__main__":
    main()
