"""
Evaluate all baselines from existing checkpoints — no re-training.

Outputs:
  outputs/baseline_comparison.json
  outputs/baseline_comparison.png   (bar chart)

Usage:
  python eval_baselines.py
"""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import json
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.dataset import get_dataloaders, build_splits
from src.baselines import (
    make_resnet50, make_efficientnet_b4, make_vit_base,
    predict_nn, build_svm_features,
)
from src.model import load_checkpoint
from src.utils import (
    set_seed, compute_and_print_metrics, save_metrics,
    topk_accuracy,
)
from sklearn.metrics import accuracy_score, f1_score
import torch


@torch.no_grad()
def eval_nn_baseline(name, model_fn, ckpt_path, test_loader, device):
    print(f"\n{'-'*50}  {name}")
    model = model_fn()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    all_preds, all_labels, all_logits = [], [], []
    for imgs, tex, col, labels in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_t   = torch.tensor(all_labels)
    top5 = topk_accuracy(logits_cat, labels_t, k=min(5, config.NUM_CLASSES))

    acc  = accuracy_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds, average="macro")
    f1_w = f1_score(all_labels, all_preds, average="weighted")

    print(f"  Top-1 : {acc:.4f}  |  Top-5 : {top5:.4f}  |  F1-macro : {f1_m:.4f}")
    return {"model": name, "top1": acc, "top5": top5, "f1_macro": f1_m, "f1_weighted": f1_w}


def eval_svm(pkl_path, x_te, y_te):
    print(f"\n{'-'*50}  SVM (HOG + Color Histogram)")
    pipe = joblib.load(pkl_path)
    X_te = build_svm_features(x_te)
    y_pred = pipe.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1_m = f1_score(y_te, y_pred, average="macro")
    f1_w = f1_score(y_te, y_pred, average="weighted")
    print(f"  Top-1 : {acc:.4f}  |  F1-macro : {f1_m:.4f}")
    return {"model": "SVM (HOG + Color)", "top1": acc, "top5": None, "f1_macro": f1_m, "f1_weighted": f1_w}


@torch.no_grad()
def eval_spicefusionnet(ckpt_path, mode, test_loader, device):
    label = "SpiceFusionNet (ours)" if mode == "fusion" else "EfficientNet-B4 image-only (Phase 1)"
    print(f"\n{'-'*50}  {label}")
    model, _, _, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    all_preds, all_labels, all_logits = [], [], []
    for imgs, tex, col, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if mode == "fusion":
            tex, col = tex.to(device), col.to(device)
            logits, _ = model.forward_fusion(imgs, tex, col)
        else:
            logits = model.forward_image(imgs)
        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_t   = torch.tensor(all_labels)
    top5 = topk_accuracy(logits_cat, labels_t, k=min(5, config.NUM_CLASSES))

    acc  = accuracy_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds, average="macro")
    f1_w = f1_score(all_labels, all_preds, average="weighted")
    print(f"  Top-1 : {acc:.4f}  |  Top-5 : {top5:.4f}  |  F1-macro : {f1_m:.4f}")
    return {"model": label, "top1": acc, "top5": top5, "f1_macro": f1_m, "f1_weighted": f1_w}


def plot_comparison(results, output_path):
    models    = [r["model"] for r in results]
    top1_vals = [r["top1"] * 100 for r in results]
    f1_vals   = [r["f1_macro"] * 100 for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, top1_vals, width, label="Top-1 Accuracy (%)", color="#4C72B0", zorder=3)
    bars2 = ax.bar(x + width/2, f1_vals,   width, label="Macro F1-Score (%)", color="#DD8452", zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Baseline Comparison — SpiceNet", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nChart saved → {output_path}")


def print_table(results):
    print(f"\n{'='*75}")
    print(f"  {'Model':<40} {'Top-1':>8}  {'Top-5':>8}  {'F1-Macro':>9}")
    print(f"{'='*75}")
    for r in results:
        top5_str = f"{r['top5']*100:7.2f}%" if r["top5"] is not None else "    N/A"
        flag = "  << ours" if "ours" in r["model"] else ""
        print(f"  {r['model']:<40} {r['top1']*100:7.2f}%  {top5_str}  {r['f1_macro']*100:8.2f}%{flag}")
    print(f"{'='*75}")


def main():
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt = config.CHECKPOINT_DIR
    out  = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Loaders — image-only for baselines, multimodal for SpiceFusionNet fusion
    _, _, test_loader_img,  x_te, y_te = get_dataloaders(multimodal=False)
    _, _, test_loader_fuse, _,    _    = get_dataloaders(multimodal=True)

    results = []

    # SVM
    results.append(eval_svm(ckpt / "svm_best.pkl", x_te, y_te))

    # ResNet-50
    results.append(eval_nn_baseline(
        "ResNet-50 (fine-tuned)",
        make_resnet50, ckpt / "resnet50_best.pth",
        test_loader_img, device,
    ))

    # EfficientNet-B4 image-only baseline
    results.append(eval_nn_baseline(
        "EfficientNet-B4 (image-only)",
        make_efficientnet_b4, ckpt / "efficientnet_b4_best.pth",
        test_loader_img, device,
    ))

    # ViT-Base
    results.append(eval_nn_baseline(
        "ViT-Base (fine-tuned)",
        make_vit_base, ckpt / "vit_base_best.pth",
        test_loader_img, device,
    ))

    # SpiceFusionNet — full fusion
    results.append(eval_spicefusionnet(
        ckpt / "best.pth", "fusion",
        test_loader_fuse, device,
    ))

    print_table(results)

    save_metrics({"baselines": results}, out / "baseline_comparison.json")
    plot_comparison(results, out / "baseline_comparison.png")


if __name__ == "__main__":
    main()
