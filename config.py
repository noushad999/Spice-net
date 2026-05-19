from pathlib import Path
import os

def _resolve_base() -> Path:
    wsl = Path("/mnt/d/SpiceNet")
    win = Path("D:/SpiceNet")
    return wsl if wsl.exists() else win

_BASE          = _resolve_base()
DATA_DIR       = _BASE / "Spice_Spectrum"
DATA_DIR_SAM   = _BASE / "Spice_Spectrum_SAM"   # SAM-preprocessed (optional)
OUTPUT_DIR     = _BASE / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR        = OUTPUT_DIR / "logs"

CLASSES = [
    "black pepper", "cardamom", "cinnamon", "cloves", "coriander",
    "cumin", "ginger", "nutmeg", "paprika", "saffron", "turmeric",
]
NUM_CLASSES = len(CLASSES)

# Hard-negative pairs (class index pairs, visually confusable)
HARD_NEG_PAIRS = [
    (4, 5),   # coriander ↔ cumin
    (8, 10),  # paprika   ↔ turmeric
    (0, 3),   # black pepper ↔ cloves
    (2, 7),   # cinnamon  ↔ nutmeg
]

# Splits — paper spec: 70 / 10 / 20
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.10
TEST_RATIO   = 0.20
RANDOM_SEED  = 42

# Image
IMG_SIZE   = 224          # resize to this before model (efficient on GPU)
IMG_FULL   = 512          # original dataset resolution
IMG_MEAN   = (0.485, 0.456, 0.406)
IMG_STD    = (0.229, 0.224, 0.225)

# Model
BACKBONE        = "efficientnet_b4"
PRETRAINED      = True
CNN_DIM         = 1792    # EfficientNet-B4 num_features
TEX_DIM         = 256     # texture branch output
COL_DIM         = 128     # color branch output
PROJ_DIM        = 128     # SupCon projection head output
DROP_RATE       = 0.4

# Texture features (LBP + GLCM)
LBP_P           = 8
LBP_R           = 1
GLCM_DISTANCES  = [1, 2]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_LEVELS     = 64
TEX_INPUT_DIM   = (LBP_P + 2) + (6 * len(GLCM_DISTANCES) * len(GLCM_ANGLES_DEG))  # 10 + 48 = 58

# Color features (HSV histogram)
HSV_H_BINS      = 36
HSV_S_BINS      = 32
HSV_V_BINS      = 32
COL_INPUT_DIM   = HSV_H_BINS + HSV_S_BINS + HSV_V_BINS  # 100

# Phase 1 — backbone pre-training
P1_EPOCHS       = 30
P1_LR           = 1e-4
P1_WARMUP       = 3
P1_WEIGHT_DECAY = 1e-4
P1_LABEL_SMOOTH = 0.1
P1_MIN_LR       = 1e-6

# Phase 2 — contrastive fine-tuning
P2_EPOCHS       = 10
P2_LR           = 5e-5
P2_TEMPERATURE  = 0.07

# Phase 3 — full fusion end-to-end
P3_EPOCHS       = 10
P3_LR           = 1e-5
P3_WEIGHT_DECAY = 1e-4
P3_LABEL_SMOOTH = 0.1
P3_ALPHA        = 0.5     # CE loss weight  (alpha*CE + (1-alpha)*SupCon)

# Shared training
BATCH_SIZE      = 32
NUM_WORKERS     = 4
GRAD_CLIP       = 1.0
PATIENCE        = 8

# Augmentation
AUG_ROTATION    = 30
AUG_SCALE       = (0.7, 1.0)
AUG_BRIGHTNESS  = 0.3
AUG_CONTRAST    = 0.3
AUG_SATURATION  = 0.3
AUG_HUE         = 0.1

# Experiment tracking (optional)
USE_WANDB       = False
WANDB_PROJECT   = "spicenet"
