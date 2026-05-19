"""
Run all 4 baselines and save comparison table.
  B1 — ResNet-50
  B2 — EfficientNet-B4 (image-only, no fusion)
  B3 — ViT-Base/16
  B4 — SVM on HOG + Color Histogram

Usage: python run_baselines.py [--epochs 20]
"""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import argparse
import json
import torch
from pathlib import Path

import config
from src.dataset import get_dataloaders, build_splits
from src.baselines import (
    make_resnet50, make_efficientnet_b4, make_vit_base,
    finetune, predict_nn, train_svm, build_svm_features,
)
from src.utils import (
    set_seed, compute_and_print_metrics, plot_confusion_matrix,
    save_metrics, measure_inference_time,
)
from sklearn.metrics import accuracy_score, f1_score
import joblib


def run_nn_baseline(name, model_fn, train_loader, val_loader, test_loader, device, epochs, ckpt_dir):
    print(f"\n{'─'*50}")
    print(f"  Baseline: {name}")
    print(f"{'─'*50}")
    model = model_fn()
    best_acc = finetune(model, train_loader, val_loader, device, epochs=epochs, name=name, ckpt_dir=ckpt_dir)

    # Load best, evaluate on test
    state = torch.load(ckpt_dir / f"{name}_best.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device)

    infer_ms  = measure_inference_time(model, test_loader, device)
    y_true, y_pred = predict_nn(model, test_loader, device)
    metrics = compute_and_print_metrics(y_true, y_pred, config.CLASSES, infer_ms=infer_ms)
    metrics["model"] = name
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(multimodal=False)
    x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(config.DATA_DIR)

    ckpt_dir   = config.CHECKPOINT_DIR
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # B1 — ResNet-50
    m, yt, yp = run_nn_baseline("resnet50", make_resnet50, train_loader, val_loader, test_loader, device, args.epochs, ckpt_dir)
    plot_confusion_matrix(yt, yp, config.CLASSES, output_dir, prefix="resnet50")
    all_results.append(m)

    # B2 — EfficientNet-B4 (image-only)
    m, yt, yp = run_nn_baseline("efficientnet_b4", make_efficientnet_b4, train_loader, val_loader, test_loader, device, args.epochs, ckpt_dir)
    plot_confusion_matrix(yt, yp, config.CLASSES, output_dir, prefix="efficientnet_b4")
    all_results.append(m)

    # B3 — ViT-Base
    m, yt, yp = run_nn_baseline("vit_base", make_vit_base, train_loader, val_loader, test_loader, device, args.epochs, ckpt_dir)
    plot_confusion_matrix(yt, yp, config.CLASSES, output_dir, prefix="vit_base")
    all_results.append(m)

    # B4 — SVM
    print(f"\n{'─'*50}")
    print("  Baseline: SVM (HOG + Color Histogram)")
    print(f"{'─'*50}")
    val_acc, svm_pipe = train_svm(x_tr, y_tr, x_val, y_val, ckpt_dir)

    print("  Extracting test features...")
    X_te = build_svm_features(x_te)
    y_pred_svm = svm_pipe.predict(X_te)
    svm_metrics = compute_and_print_metrics(y_te, y_pred_svm, config.CLASSES)
    svm_metrics["model"] = "svm_hog_color"
    plot_confusion_matrix(y_te, y_pred_svm.tolist(), config.CLASSES, output_dir, prefix="svm")
    all_results.append(svm_metrics)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  {'Model':<22} {'Top-1 Acc':>10} {'Macro F1':>10} {'Infer(ms)':>10}")
    print(f"{'='*65}")
    for r in all_results:
        ms = r.get("inference_ms", -1)
        ms_str = f"{ms:.2f}" if ms >= 0 else "N/A"
        print(f"  {r['model']:<22} {r['top1_accuracy']:>10.4f} {r['f1_macro']:>10.4f} {ms_str:>10}")
    print(f"{'='*65}")

    save_metrics({"baselines": all_results}, output_dir / "baseline_comparison.json")
    print(f"\nResults saved → {output_dir}/baseline_comparison.json")


if __name__ == "__main__":
    main()
