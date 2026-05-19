# test_sanity.py — Unit / Sanity Tests

## Overview

Lightweight test suite that verifies feature extraction shapes and model forward pass
dimensions are correct. Run this after any changes to `config.py`, `src/features.py`,
or `src/model.py` to catch regressions immediately.

```bash
python test_sanity.py
```

---

## Test Structure

Uses a simple custom test runner (no pytest dependency):

```python
PASS, FAIL = "[PASS]", "[FAIL]"
results = []

def run(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"{PASS} {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL} {name} -> {e}")
        traceback.print_exc()
```

Each test is a function that raises an `AssertionError` on failure.
At the end, exit code 0 = all passed, 1 = any failure.

---

## Test Fixture

```python
rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
```

A random 512×512×3 uint8 image used for all feature extraction tests.
Deterministic seed ensures consistent test behavior.

---

## Feature Extraction Tests

### `test_lbp()` — LBP output shape
```python
def test_lbp():
    v = features.extract_lbp(img)
    assert v.shape == (config.LBP_P + 2,), v.shape   # (10,)
    assert v.dtype == np.float32
```
Checks output is 10-d float32.

### `test_glcm()` — GLCM output shape
```python
def test_glcm():
    v = features.extract_glcm(img)
    expected = 6 * len(config.GLCM_DISTANCES) * len(config.GLCM_ANGLES_DEG)
    assert v.shape == (expected,)   # 6×2×4 = 48
```
Checks output is 48-d.

### `test_texture()` — Combined texture shape
```python
def test_texture():
    v = features.extract_texture(img)
    assert v.shape == (config.TEX_INPUT_DIM,)   # 58 = 10 + 48
```

### `test_hsv()` — HSV histogram shape
```python
def test_hsv():
    v = features.extract_hsv(img)
    assert v.shape == (config.COL_INPUT_DIM,)   # 100 = 36+32+32
```

### `test_all()` — Combined extraction
```python
def test_all():
    tex, col = features.extract_all(img)
    assert tex.shape == (58,)
    assert col.shape == (100,)
```

---

## Model Forward Tests

All model tests use `pretrained=False` to avoid downloading weights during testing.

### `test_model_image_forward()` — Phase 1 forward
```python
def test_model_image_forward():
    net = SpiceFusionNet(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    logits = net.forward_image(x)
    assert logits.shape == (2, 11)   # (batch=2, num_classes=11)
```

### `test_model_contrastive_forward()` — Phase 2 forward + L2 norm check
```python
def test_model_contrastive_forward():
    proj = net.forward_contrastive(x)
    assert proj.shape == (2, 128)
    norms = proj.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)   # unit sphere
```
Verifies L2 normalization is applied — all projections should have norm = 1.0.

### `test_model_fusion_forward()` — Phase 3 forward
```python
def test_model_fusion_forward():
    x   = torch.randn(2, 3, 224, 224)
    tex = torch.randn(2, 58)
    col = torch.randn(2, 100)
    logits, proj = net.forward_fusion(x, tex, col)
    assert logits.shape == (2, 11)
    assert proj.shape   == (2, 128)
```

### `test_attention_sum_to_one()` — Gate verification
```python
def test_attention_sum_to_one():
    fusion = AttentionFusion(1792, 256, 128).eval()
    f_cnn = torch.randn(4, 1792)
    f_tex = torch.randn(4, 256)
    f_col = torch.randn(4, 128)
    cat = torch.cat([f_cnn, f_tex, f_col], dim=1)
    gates = fusion.gate(cat)
    assert gates.shape == (4, 3)
    assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1e-5)
```
Verifies the softmax gate sums to 1.0 for each sample.

### `test_config_dims_consistent()` — Config sanity
```python
def test_config_dims_consistent():
    assert config.TEX_INPUT_DIM == 58
    assert config.COL_INPUT_DIM == 100
    assert config.NUM_CLASSES == len(config.CLASSES) == 11
```
Ensures `config.py` constants are internally consistent.

---

## Expected Output

```
[PASS] config dims consistent
[PASS] extract_lbp shape
[PASS] extract_glcm shape
[PASS] extract_texture shape
[PASS] extract_hsv shape
[PASS] extract_all shapes
[PASS] AttentionFusion gates sum-to-1
[PASS] model.forward_image shape
[PASS] model.forward_contrastive L2-norm
[PASS] model.forward_fusion shape

=== 10 passed, 0 failed ===
```

---

## When to Run

- After modifying `config.py` (especially dimension constants)
- After changing feature extraction in `src/features.py`
- After changing model architecture in `src/model.py`
- As a pre-training sanity check before a long training run
