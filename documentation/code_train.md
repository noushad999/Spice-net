# train.py — Training Entry Point

## Overview

The main script to run the 3-phase training pipeline.
It is a thin orchestration layer — the actual training logic lives in `src/trainer.py`.

---

## Command-Line Arguments

```
python train.py                          # Run all 3 phases (recommended)
python train.py --phase 1               # Only Phase 1
python train.py --phase 2               # Only Phase 2 (requires p1_best.pth)
python train.py --phase 3               # Only Phase 3 (requires p2_last.pth)
python train.py --multimodal            # Enable hand-crafted features in Phase 1/2
python train.py --data_dir /path/to/X   # Override dataset directory (e.g., SAM variant)
```

| Argument | Default | Description |
|---|---|---|
| `--phase` | `0` | `0` = all phases, `1/2/3` = single phase |
| `--multimodal` | `False` | Enable texture+color features (always on in Phase 3) |
| `--data_dir` | `config.DATA_DIR` | Path to dataset root |

---

## Execution Flow

```python
def main():
    # 1. Parse args
    args = parser.parse_args()

    # 2. Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)

    # 3. Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Determine data directory
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR

    # 5. Phase 3 always needs multimodal features
    multimodal = args.multimodal or args.phase == 3

    # 6. Build DataLoaders
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        multimodal=multimodal, data_dir=data_dir
    )

    # 7. Create model and trainer
    model   = SpiceFusionNet()
    trainer = PhaseTrainer(model, device, config.CHECKPOINT_DIR)

    # 8. Determine which phases to run
    run_phases = [args.phase] if args.phase in (1,2,3) else [1, 2, 3]

    # 9. Execute phases in order
    for phase in run_phases:
        if phase == 1:
            h1 = trainer.phase1(train_loader, val_loader)
            plot_training_curves(h1, config.OUTPUT_DIR, prefix="p1")
        elif phase == 2:
            trainer.phase2(train_loader)
        elif phase == 3:
            # Auto-switch to multimodal loaders if needed
            if not multimodal:
                train_loader, val_loader, test_loader, _, _ = get_dataloaders(multimodal=True)
            h3 = trainer.phase3(train_loader, val_loader)
            plot_training_curves(h3, config.OUTPUT_DIR, prefix="p3")
```

---

## Key Design Decisions

**Phase 3 always multimodal:**
Phase 3 trains all branches including texture and color.
If the user runs `python train.py --phase 3` without `--multimodal`,
the script auto-detects this and re-creates multimodal DataLoaders.

**DataLoader for Phase 1+2 vs Phase 3:**
- Phase 1+2: `multimodal=False` → fast loading (no feature extraction per image)
- Phase 3: `multimodal=True` → slower (LBP+GLCM+HSV per image) but necessary

**No test evaluation:**
`train.py` only trains — it does not evaluate on the test set.
Test evaluation is handled by `evaluate.py` separately.
This enforces the test set as a strictly held-out final benchmark.

---

## W&B Integration

```python
try:
    import wandb
    if config.USE_WANDB:
        wandb.init(project="spicenet", config=vars(config))
except ImportError:
    pass
```

Optional — if W&B is installed and `USE_WANDB=True`, experiment tracking is enabled.
If not installed, the script continues without it.

---

## Checkpoint Flow

```
Phase 1 → writes: p1_best.pth, p1_last.pth
Phase 2 → reads: p1_best.pth → writes: p2_last.pth
Phase 3 → reads: p2_last.pth → writes: best.pth, last.pth
```

Each phase reads the output of the previous. If running individual phases,
the required input checkpoint must exist.

---

## Outputs After Full Run

```
outputs/
├── checkpoints/
│   ├── p1_best.pth    ← Phase 1 best model
│   ├── p1_last.pth    ← Phase 1 last epoch
│   ├── p2_last.pth    ← Phase 2 final
│   ├── best.pth       ← Phase 3 best model (primary output)
│   └── last.pth       ← Phase 3 last epoch
├── p1_training_curves.png
└── p3_training_curves.png
```
