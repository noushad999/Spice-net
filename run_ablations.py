"""
5 Ablation Studies (paper spec):
  A1 — Component ablation:     image-only vs +texture vs +color vs full fusion
  A2 — Contrastive loss:       with vs without Phase 2 SupCon
  A3 — Augmentation:           none vs standard vs spice-specific
  A4 — Backbone:               B0 / B2 / B4 vs ResNet-50 vs MobileNetV3
  A5 — Background removal:     raw images vs SAM-preprocessed

Usage:
  python run_ablations.py --ablation A1
  python run_ablations.py --ablation all
"""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from src.dataset import get_dataloaders, build_splits, SpiceDataset, get_val_transform
from src.model import SpiceFusionNet
from src.trainer import PhaseTrainer
from src.baselines import finetune, make_resnet50, predict_nn
from src.utils import set_seed, compute_and_print_metrics, save_metrics

EPOCHS_ABLATION = 8    # kept low to avoid Blackwell CUDA instability on long runs


def _quick_finetune(model, train_loader, val_loader, test_loader, device, name, epochs=EPOCHS_ABLATION):
    from src.baselines import finetune, predict_nn
    finetune(model, train_loader, val_loader, device, epochs=epochs,
             name=name, ckpt_dir=config.CHECKPOINT_DIR)
    state = torch.load(config.CHECKPOINT_DIR / f"{name}_best.pth", map_location=device)
    model.load_state_dict(state); model.to(device)
    y_true, y_pred = predict_nn(model, test_loader, device)
    m = compute_and_print_metrics(y_true, y_pred, config.CLASSES)
    m["name"] = name
    return m


# ── A1: Component ablation ────────────────────────────────────────────────────

def ablation_A1(device):
    """image-only vs +texture vs +color vs full fusion."""
    print("\n=== A1: Component Ablation ===")
    results = []

    # Image-only: EfficientNet-B4
    train_l, val_l, te_l, _, _ = get_dataloaders(multimodal=False)
    m = _quick_finetune(
        timm.create_model("efficientnet_b4", pretrained=True, num_classes=config.NUM_CLASSES),
        train_l, val_l, te_l, device, "a1_image_only",
    )
    results.append(m)

    # Full fusion (Phase 1 + 3 only, skip Phase 2 for speed)
    train_mm, val_mm, te_mm, _, _ = get_dataloaders(multimodal=True)
    model = SpiceFusionNet()
    trainer = PhaseTrainer(model, device, config.CHECKPOINT_DIR)
    trainer.phase1(train_l, val_l)
    h3 = trainer.phase3(train_mm, val_mm)

    y_true, y_pred, _ = _eval_fusion(model, te_mm, device)
    m = compute_and_print_metrics(y_true, y_pred, config.CLASSES)
    m["name"] = "a1_full_fusion"
    results.append(m)

    return results


# ── A2: With vs without SupCon ────────────────────────────────────────────────

def ablation_A2(device):
    print("\n=== A2: Contrastive Loss Ablation ===")
    results = []

    train_l, val_l, te_l, _, _ = get_dataloaders(multimodal=False)
    train_mm, val_mm, te_mm, _, _ = get_dataloaders(multimodal=True)

    # Without SupCon (Phase 1 → Phase 3 directly)
    model = SpiceFusionNet()
    trainer = PhaseTrainer(model, device, config.CHECKPOINT_DIR)
    trainer.phase1(train_l, val_l)
    trainer.phase3(train_mm, val_mm)
    y_true, y_pred, _ = _eval_fusion(model, te_mm, device)
    m = compute_and_print_metrics(y_true, y_pred, config.CLASSES)
    m["name"] = "a2_no_supcon"; results.append(m)

    # With SupCon (all 3 phases)
    model2 = SpiceFusionNet()
    trainer2 = PhaseTrainer(model2, device, config.CHECKPOINT_DIR)
    trainer2.phase1(train_l, val_l)
    trainer2.phase2(train_l)
    trainer2.phase3(train_mm, val_mm)
    y_true, y_pred, _ = _eval_fusion(model2, te_mm, device)
    m = compute_and_print_metrics(y_true, y_pred, config.CLASSES)
    m["name"] = "a2_with_supcon"; results.append(m)

    return results


# ── A3: Augmentation ablation ─────────────────────────────────────────────────

def _make_loader_with_aug(aug_tf, multimodal=False):
    x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(config.DATA_DIR)
    val_tf = get_val_transform()
    tr_ds  = SpiceDataset(x_tr, y_tr, aug_tf, multimodal)
    val_ds = SpiceDataset(x_val, y_val, val_tf, multimodal)
    te_ds  = SpiceDataset(x_te, y_te, val_tf, multimodal)
    mk = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True, drop_last=True, **mk),
            DataLoader(val_ds, shuffle=False, **mk),
            DataLoader(te_ds, shuffle=False, **mk))


def ablation_A3(device):
    print("\n=== A3: Augmentation Ablation ===")
    results = []
    norm = A.Compose([A.Resize(224, 224), A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD), ToTensorV2()])

    # No augmentation
    tr, val, te = _make_loader_with_aug(norm)
    m = _quick_finetune(
        timm.create_model("efficientnet_b4", pretrained=True, num_classes=config.NUM_CLASSES),
        tr, val, te, device, "a3_no_aug",
    )
    results.append(m)

    # Standard augmentation
    std_aug = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD), ToTensorV2(),
    ])
    tr, val, te = _make_loader_with_aug(std_aug)
    m = _quick_finetune(
        timm.create_model("efficientnet_b4", pretrained=True, num_classes=config.NUM_CLASSES),
        tr, val, te, device, "a3_std_aug",
    )
    results.append(m)

    # Spice-specific augmentation (full from dataset.py)
    from src.dataset import get_train_transform
    tr, val, te = _make_loader_with_aug(get_train_transform())
    m = _quick_finetune(
        timm.create_model("efficientnet_b4", pretrained=True, num_classes=config.NUM_CLASSES),
        tr, val, te, device, "a3_spice_aug",
    )
    results.append(m)

    return results


# ── A4: Backbone ablation ─────────────────────────────────────────────────────

def ablation_A4(device):
    print("\n=== A4: Backbone Ablation ===")
    results = []
    train_l, val_l, te_l, _, _ = get_dataloaders(multimodal=False)

    backbones = [
        ("efficientnet_b0", "a4_eff_b0"),
        ("efficientnet_b2", "a4_eff_b2"),
        ("efficientnet_b4", "a4_eff_b4"),
        ("resnet50",        "a4_resnet50"),
        ("mobilenetv3_large_100", "a4_mobilenetv3"),
    ]
    for bb, name in backbones:
        m = _quick_finetune(
            timm.create_model(bb, pretrained=True, num_classes=config.NUM_CLASSES),
            train_l, val_l, te_l, device, name,
        )
        results.append(m)

    return results


# ── A5: SAM background removal ────────────────────────────────────────────────

def ablation_A5(device):
    print("\n=== A5: Background Removal (SAM) Ablation ===")
    results = []

    for name, data_dir in [("a5_raw", config.DATA_DIR), ("a5_sam", config.DATA_DIR_SAM)]:
        if not data_dir.exists():
            print(f"  Skipping {name}: {data_dir} not found. Run sam_preprocess.py first.")
            continue
        train_l, val_l, te_l, _, _ = get_dataloaders(multimodal=False, data_dir=data_dir)
        m = _quick_finetune(
            timm.create_model("efficientnet_b4", pretrained=True, num_classes=config.NUM_CLASSES),
            train_l, val_l, te_l, device, name,
        )
        results.append(m)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_fusion(model, loader, device):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    for imgs, tex, col, labels in loader:
        imgs, tex, col = imgs.to(device), tex.to(device), col.to(device)
        logits, _ = model.forward_fusion(imgs, tex, col)
        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
    import torch as T
    return all_labels, all_preds, T.cat(all_logits)


def print_table(results: list, title: str):
    print(f"\n{'='*55}  {title}")
    print(f"  {'Name':<25} {'Top-1':>8} {'Macro F1':>10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r.get('name','?'):<25} {r['top1_accuracy']:>8.4f} {r['f1_macro']:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="all",
                        choices=["A1","A2","A3","A4","A5","all"])
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = {"A1": ablation_A1, "A2": ablation_A2, "A3": ablation_A3,
                 "A4": ablation_A4, "A5": ablation_A5}

    run = list(ablations.keys()) if args.ablation == "all" else [args.ablation]
    all_results = {}

    for key in run:
        res = ablations[key](device)
        all_results[key] = res
        print_table(res, key)

    save_metrics(all_results, output_dir / "ablation_results.json")
    print(f"\nAblation results → {output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
