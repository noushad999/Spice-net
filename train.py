"""
SpiceNet — 3-Phase Training
  Phase 1: EfficientNet-B4 backbone pre-training (image-only, CE loss)
  Phase 2: Contrastive fine-tuning (SupCon loss on backbone)
  Phase 3: Full fusion end-to-end (CE + SupCon, all branches active)

Usage:
  python train.py                          # full 3-phase
  python train.py --phase 1               # only Phase 1
  python train.py --phase 3 --multimodal  # Phase 3 with texture/color features
  python train.py --data_dir /mnt/d/SpiceNet/Spice_Spectrum_SAM  # SAM-preprocessed
"""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import argparse
import torch
from pathlib import Path

import config
from src.dataset import get_dataloaders
from src.model import SpiceFusionNet
from src.trainer import PhaseTrainer
from src.utils import set_seed, plot_training_curves

try:
    import wandb
    if config.USE_WANDB:
        wandb.init(project=config.WANDB_PROJECT, config=vars(config))
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=0,   help="0=all phases, 1/2/3=single phase")
    parser.add_argument("--multimodal", action="store_true",    help="Enable texture+color branches (Phase 3)")
    parser.add_argument("--data_dir",   type=str, default=None, help="Override dataset directory (e.g., SAM preprocessed)")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR

    # Phase 3 always needs multimodal data
    multimodal = args.multimodal or args.phase == 3

    print(f"Loading data from: {data_dir}")
    print(f"Multimodal features: {multimodal}")
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        multimodal=multimodal,
        data_dir=data_dir,
    )
    print(f"  train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}")

    model = SpiceFusionNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"SpiceFusionNet params: {total:,}")

    trainer = PhaseTrainer(model, device, config.CHECKPOINT_DIR)

    run_phases = [args.phase] if args.phase in (1, 2, 3) else [1, 2, 3]

    for phase in run_phases:
        if phase == 1:
            h1 = trainer.phase1(train_loader, val_loader)
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            plot_training_curves(h1, config.OUTPUT_DIR, prefix="p1")
        elif phase == 2:
            trainer.phase2(train_loader)
        elif phase == 3:
            if not multimodal:
                print("Switching to multimodal loaders for Phase 3 (texture+color features)...")
                train_loader, val_loader, test_loader, _, _ = get_dataloaders(
                    multimodal=True, data_dir=data_dir)
                multimodal = True
            h3 = trainer.phase3(train_loader, val_loader)
            plot_training_curves(h3, config.OUTPUT_DIR, prefix="p3")

    print("\nTraining complete. Run evaluate.py for test metrics.")


if __name__ == "__main__":
    main()
