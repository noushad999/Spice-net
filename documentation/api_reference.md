# API Reference

All public modules live under `src/`. This file documents every class and function signature.

---

## `src/model.py`

### `SpiceFusionNet`

```python
class SpiceFusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        backbone: str = config.BACKBONE,
        texture_dim: int = config.TEXTURE_DIM,
        color_dim: int = config.COLOR_DIM,
        proj_dim: int = config.PROJ_DIM,
    )
```

Main multi-modal classification network. See [architecture.md](architecture.md) for details.

**Methods:**

| Method | Signature | Returns | Phase |
|---|---|---|---|
| `forward_image` | `(x: Tensor[B,3,224,224])` | `logits [B, 11]` | 1 |
| `forward_contrastive` | `(x: Tensor)` | `embed [B, 128]` (L2-norm) | 2 |
| `forward_fusion` | `(x, tex [B,58], col [B,100])` | `(logits [B,11], proj [B,128])` | 3 |
| `forward` | `(x, tex=None, col=None)` | auto-selects mode | any |

---

### `save_checkpoint`

```python
def save_checkpoint(
    model: SpiceFusionNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    path: str,
) -> None
```

Saves model weights + optimizer state + metadata to `path`.

---

### `load_checkpoint`

```python
def load_checkpoint(
    model: SpiceFusionNet,
    path: str,
    device: str = "cpu",
) -> tuple[SpiceFusionNet, dict]
```

Returns `(model_with_weights, metadata_dict)`. `metadata_dict` contains `epoch`, `best_val_acc`,
and `config` keys.

---

### `_MLP`

```python
class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.3)
```

Shallow MLP: `Linear → BN → ReLU → Dropout → ... → Linear → BN → ReLU`.

---

### `AttentionFusion`

```python
class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim: int, tex_dim: int, col_dim: int)
    def forward(self, cnn: Tensor, tex: Tensor, col: Tensor) -> Tensor
```

Computes softmax gate weights and returns weighted sum of all three branch features.

---

## `src/dataset.py`

### `SpiceDataset`

```python
class SpiceDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform,
        multimodal: bool = False,
        full_size: int = config.FULL_SIZE,
    )
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, int]
    # returns: (image, texture_58d, color_100d, label)
```

If `multimodal=False`, texture and color are zero tensors of the correct shape.

---

### `get_dataloaders`

```python
def get_dataloaders(
    data_dir: str = config.DATA_DIR,
    batch_size: int = config.P1_BATCH_SIZE,
    multimodal: bool = False,
    seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]
# returns: (train_loader, val_loader, test_loader)
```

---

### `build_splits`

```python
def build_splits(
    data_dir: str,
    seed: int = config.RANDOM_SEED,
) -> tuple[list, list, list]
# returns: (train_samples, val_samples, test_samples)
# Each sample is (path: str, label: int)
```

---

### `get_train_transform` / `get_val_transform`

```python
def get_train_transform(image_size: int = config.IMAGE_SIZE) -> albumentations.Compose
def get_val_transform(image_size: int = config.IMAGE_SIZE) -> albumentations.Compose
```

---

## `src/features.py`

### `extract_lbp`

```python
def extract_lbp(
    image_rgb: np.ndarray,    # H×W×3, uint8
    n_points: int = 24,
    radius: int = 3,
) -> np.ndarray               # shape (10,), float32
```

Computes uniform LBP histogram. Normalizes to sum-to-1.

---

### `extract_glcm`

```python
def extract_glcm(
    image_rgb: np.ndarray,
    distances: list[int] = [1, 3],
    angles: list[float] = [0, π/4, π/2, 3π/4],
) -> np.ndarray               # shape (48,), float32
```

Computes GLCM with 6 properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
across all distance×angle combinations. Flattened and concatenated.

---

### `extract_texture`

```python
def extract_texture(image_rgb: np.ndarray) -> np.ndarray  # shape (58,)
```

Convenience wrapper: `np.concatenate([extract_lbp(img), extract_glcm(img)])`.

---

### `extract_hsv`

```python
def extract_hsv(
    image_rgb: np.ndarray,
    h_bins: int = 36,
    s_bins: int = 32,
    v_bins: int = 32,
) -> np.ndarray               # shape (100,), float32
```

Converts to HSV, computes per-channel histogram, normalizes each to sum-to-1, concatenates.

---

### `extract_all`

```python
def extract_all(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]
# returns: (texture_58d, color_100d)
```

---

## `src/trainer.py`

### `PhaseTrainer`

```python
class PhaseTrainer:
    def __init__(
        self,
        model: SpiceFusionNet,
        device: torch.device,
        output_dir: str = config.OUTPUT_DIR,
    )

    def phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.P1_EPOCHS,
    ) -> SpiceFusionNet

    def phase2(
        self,
        train_loader: DataLoader,
        epochs: int = config.P2_EPOCHS,
    ) -> SpiceFusionNet

    def phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.P3_EPOCHS,
    ) -> SpiceFusionNet
```

Each phase method mutates the model in-place and also returns it for chaining.
Checkpoints are saved automatically during training.

---

## `src/losses.py`

### `SupConLoss`

```python
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07)
    def forward(
        self,
        features: Tensor,   # (B, proj_dim), L2-normalized
        labels: Tensor,     # (B,), int class indices
    ) -> Tensor             # scalar loss
```

Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
Positive pairs: same-class samples within the batch (excluding self).

---

### `CombinedLoss`

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 0.07)
    def forward(
        self,
        logits: Tensor,     # (B, num_classes)
        proj: Tensor,       # (B, proj_dim), L2-normalized
        labels: Tensor,     # (B,)
    ) -> Tensor             # scalar: alpha*CE + (1-alpha)*SupCon
```

---

## `src/utils.py`

### `set_seed`

```python
def set_seed(seed: int = config.RANDOM_SEED) -> None
```

Sets seeds for `random`, `numpy`, `torch`, and `torch.cuda`. Also sets `torch.backends.cudnn.deterministic=True`.

---

### `topk_accuracy`

```python
def topk_accuracy(
    outputs: Tensor,    # (B, num_classes) logits
    targets: Tensor,    # (B,) int labels
    k: int = 5,
) -> float              # percentage (0–100)
```

---

### `measure_inference_time`

```python
def measure_inference_time(
    model: SpiceFusionNet,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 20,
) -> float              # milliseconds per image
```

Warms up GPU, then averages over `n_batches` batches using CUDA events for precise timing.

---

### `plot_training_curves`

```python
def plot_training_curves(
    history: dict,       # keys: "train_loss", "val_loss", "train_acc", "val_acc", "lr"
    save_path: str,
) -> None
```

Saves a 3-subplot figure (loss, accuracy, learning rate) to `save_path`.

---

### `plot_confusion_matrix`

```python
def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str,
) -> None
```

Saves a side-by-side raw count + normalized confusion matrix.

---

### `compute_and_print_metrics`

```python
def compute_and_print_metrics(
    y_true: list[int],
    y_pred: list[int],
    top5_acc: float,
    class_names: list[str],
) -> dict
```

Prints and returns a metrics dict with `top1_accuracy`, `top5_accuracy`, `f1_macro`,
`f1_weighted`, and `per_class`.

---

### `save_metrics`

```python
def save_metrics(metrics: dict, path: str) -> None
```

Dumps metrics dict to a JSON file.

---

## `src/gradcam.py`

### `GradCAM`

```python
class GradCAM:
    def __init__(self, model: SpiceFusionNet, target_layer: nn.Module)
    def generate(self, x: Tensor) -> np.ndarray   # returns heatmap (H, W), float32
    def __del__(self)                              # removes hooks
```

Registers forward and backward hooks on `target_layer`. Computes heatmap via global average pooled
gradients of the highest-scoring class.

---

### `visualize_gradcam`

```python
def visualize_gradcam(
    model: SpiceFusionNet,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    n_wrong: int = 8,
    n_correct: int = 8,
) -> None
```

Finds misclassified samples (up to `n_wrong`) and random correct samples (up to `n_correct`),
generates Grad-CAM overlays, and saves a grid image to `save_path`.

---

## `src/baselines.py`

### Model Factories

```python
def make_resnet50(num_classes: int = 11) -> nn.Module
def make_efficientnet_b4(num_classes: int = 11) -> nn.Module
def make_vit_base(num_classes: int = 11) -> nn.Module
```

All return pretrained timm models with the classifier head replaced for `num_classes`.

---

### `finetune`

```python
def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    device: torch.device = ...,
    save_path: str = ...,
) -> nn.Module
```

Generic fine-tuning loop with CE loss, AdamW optimizer, and early stopping (patience=5).

---

### `build_svm_features`

```python
def build_svm_features(
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]   # (X, y)
```

Extracts HOG + HSV color histogram features for all images in the loader.

---

### `train_svm`

```python
def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = ...,
) -> sklearn.pipeline.Pipeline
```

Fits `StandardScaler → PCA(256) → LinearSVC` and saves the pipeline to `save_path`.

---

## `config.py`

Key constants (not a class — top-level module variables):

```python
CLASSES: list[str]          # 11 class names in order (index = label)
NUM_CLASSES: int            # 11
DATA_DIR: str               # path to Spice_Spectrum/
DATA_DIR_SAM: str           # path to Spice_Spectrum_SAM/
OUTPUT_DIR: str             # outputs/
CHECKPOINT_DIR: str         # outputs/checkpoints/
LOG_DIR: str                # outputs/logs/
RANDOM_SEED: int            # 42
IMAGE_SIZE: int             # 224
FULL_SIZE: int              # 512
BACKBONE: str               # "efficientnet_b4"
TEXTURE_DIM: int            # 256
COLOR_DIM: int              # 128
PROJ_DIM: int               # 128
HARD_NEGATIVES: list        # [(class_idx_a, class_idx_b), ...]
```
