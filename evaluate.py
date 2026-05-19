"""
Full evaluation: Top-1/5 accuracy, F1, confusion matrix, inference time, Grad-CAM.

Usage:
  python evaluate.py                           # evaluate best.pth (Phase 3 fusion)
  python evaluate.py --ckpt outputs/checkpoints/p1_best.pth --mode image
  python evaluate.py --gradcam                 # also generate Grad-CAM plots
"""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import argparse
import torch

import config
from src.dataset import get_dataloaders, build_splits
from src.model import load_checkpoint
from src.utils import (
    set_seed, compute_and_print_metrics, plot_confusion_matrix,
    save_metrics, measure_inference_time, topk_accuracy,
)
from src.gradcam import visualize_gradcam


@torch.no_grad()
def predict(model, loader, device, mode="fusion"):
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []

    for imgs, tex, col, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if mode == "fusion":
            tex, col = tex.to(device), col.to(device)
            logits, _ = model.forward_fusion(imgs, tex, col)
        else:
            logits = model.forward_image(imgs)

        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds, torch.cat(all_logits, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=str(config.CHECKPOINT_DIR / "best.pth"))
    parser.add_argument("--mode",    default="fusion", choices=["fusion", "image"])
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from pathlib import Path
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    multimodal = args.mode == "fusion"

    print(f"Checkpoint : {args.ckpt}")
    print(f"Mode       : {args.mode}")
    model, epoch, best_val_acc, _ = load_checkpoint(args.ckpt, device)
    print(f"Trained for {epoch} epochs | best val acc: {best_val_acc:.4f}")

    _, _, test_loader, x_te, y_te = get_dataloaders(
        multimodal=multimodal, data_dir=data_dir)

    # Inference time
    infer_ms = measure_inference_time(model, test_loader, device)

    # Predictions
    print(f"\nInference on {len(test_loader.dataset)} test samples...")
    y_true, y_pred, logits_all = predict(model, test_loader, device, mode=args.mode)

    # Top-5
    y_true_t = torch.tensor(y_true)
    top5 = topk_accuracy(logits_all, y_true_t, k=min(5, config.NUM_CLASSES))

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.mode}"

    metrics = compute_and_print_metrics(
        y_true, y_pred, config.CLASSES,
        top5_acc=top5,
        infer_ms=infer_ms,
    )
    save_metrics(metrics, output_dir / f"{prefix}_test_metrics.json")
    plot_confusion_matrix(y_true, y_pred, config.CLASSES, output_dir, prefix=prefix)

    # Grad-CAM
    if args.gradcam:
        # Collect misclassified samples first, then random correct ones
        wrong  = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred != l]
        right  = [(p, l) for p, l, pred in zip(x_te, y_te, y_pred) if pred == l]
        samples = wrong[:8] + right[:8]
        paths, labels = zip(*samples) if samples else ([], [])
        visualize_gradcam(
            model, list(paths), list(labels), config.CLASSES,
            device, output_dir / f"{prefix}_gradcam.png",
            mode=args.mode,
        )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
