# src/baselines.py — Comparison Baseline Models

## Overview

Implements four baseline models for paper comparison tables:
- **B1** — ResNet-50 (classic deep CNN)
- **B2** — EfficientNet-B4 image-only (same backbone as SpiceFusionNet, but no fusion)
- **B3** — ViT-Base/16 (Vision Transformer)
- **B4** — SVM on HOG + Color Histogram (classical machine learning)

All neural baselines share a common fine-tuning loop.

---

## Model Factories

```python
def make_resnet50(num_classes=11):
    return timm.create_model("resnet50", pretrained=True, num_classes=11)

def make_efficientnet_b4(num_classes=11):
    return timm.create_model("efficientnet_b4", pretrained=True, num_classes=11)

def make_vit_base(num_classes=11):
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=11)
```

All use `timm` (PyTorch Image Models library) with ImageNet pretrained weights.
`num_classes=11` replaces the original 1000-class head with an 11-class head.

**Why these baselines?**
- ResNet-50: standard convolutional baseline
- EfficientNet-B4: same backbone as SpiceFusionNet — isolates the benefit of fusion
- ViT-Base: transformer-based architecture — different inductive bias than CNNs
- SVM: traditional ML ceiling — shows how much deep learning helps

---

## Function: `finetune()` — Generic Neural Training Loop

```python
def finetune(model, train_loader, val_loader, device,
             epochs=20, lr=1e-4, name="baseline", ckpt_dir=...):
```

Shared training loop for all three neural baselines:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

for epoch in range(1, epochs+1):
    # Train loop
    for imgs, tex, col, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)   # model(imgs) uses default forward
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validate every epoch
    val_acc = evaluate_nn(model, val_loader, device)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{name}_best.pth")
```

Note: `tex` and `col` from the DataLoader are ignored here — baselines are image-only.

### Returns: `best_acc` (float)

---

## Function: `evaluate_nn()` — Accuracy Measurement

```python
@torch.no_grad()
def evaluate_nn(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, tex, col, labels in loader:
        preds = model(imgs.to(device)).argmax(1)
        correct += (preds == labels.to(device)).sum().item()
        total   += labels.size(0)
    return correct / total
```

Simple top-1 accuracy. Used internally by `finetune()` for validation.

---

## Function: `predict_nn()` — Get All Predictions

```python
@torch.no_grad()
def predict_nn(model, loader, device):
    all_preds, all_labels = [], []
    for imgs, tex, col, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds
```

Returns all ground-truth labels and predictions for the full dataset.
Used after training to compute confusion matrices and F1 scores.

---

## SVM Baseline (B4)

### Feature Extraction

#### HOG Features: `_hog_features(img_bgr)`

```python
def _hog_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    hog = cv2.HOGDescriptor(
        winSize=(224,224), blockSize=(16,16), blockStride=(8,8),
        cellSize=(8,8), nbins=9
    )
    return hog.compute(gray).ravel()
```

**HOG = Histogram of Oriented Gradients**
- Divides image into 8×8 pixel cells
- Computes gradient orientation histogram per cell (9 bins = 0–180°)
- Groups cells into 16×16 blocks and normalizes
- Output: high-dimensional edge/gradient descriptor

HOG captures shape information (edges, contours of spice structures).

#### Color Histogram: `_color_hist(img_bgr)`

```python
def _color_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).ravel()
    feat = np.concatenate([h, s, v])
    return feat / (feat.sum() + 1e-8)   # L1 normalize
```

Same HSV histogram as SpiceFusionNet's color branch — 100-d vector.
Allows fair comparison: the SVM has access to color information just like our model.

#### Combined SVM Features: `build_svm_features(paths)`

```python
def build_svm_features(paths):
    for p in paths:
        img = cv2.imread(p)
        img = cv2.resize(img, (224, 224))
        f = np.concatenate([_hog_features(img), _color_hist(img)])
        feats.append(f)
    return np.stack(feats)   # (N, HOG_dim + 100)
```

Final feature vector per image = HOG + Color Histogram concatenated.
Handles failed image loads gracefully (replaces with zeros).

### SVM Pipeline: `train_svm(x_train, y_train, x_val, y_val)`

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),          # normalize feature scale
    ("pca",    PCA(n_components=256)),     # reduce to 256-d
    ("svm",    LinearSVC(C=1.0, max_iter=2000)),
])
pipe.fit(X_tr, y_train)
```

**Why `StandardScaler`?**
HOG and color histogram values have very different scales.
StandardScaler makes all features zero-mean, unit-variance.

**Why `PCA(256)`?**
HOG feature dimension is very high (~10,000+). PCA reduces to 256 principal
components while retaining most variance. Also prevents LinearSVC from overfitting.

**Why `LinearSVC` not `SVC(kernel='rbf')`?**
LinearSVC scales to large datasets (O(N) training time vs O(N²-N³) for kernel SVM).
With PCA-256, the feature space is already well-structured, so a linear boundary works.

Saves the trained pipeline to `svm_best.pkl` using joblib for later inference.

### Returns: `(val_acc, svm_pipe)`
