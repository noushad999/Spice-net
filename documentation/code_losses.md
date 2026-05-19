# src/losses.py — Loss Functions

## Overview

Two loss functions are defined here:
1. `SupConLoss` — Supervised Contrastive Loss (Phase 2)
2. `CombinedLoss` — 0.5×CE + 0.5×SupCon (Phase 3)

---

## Class: `SupConLoss`

**Paper:** Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

### Core Idea

Standard CrossEntropy only cares whether the top prediction is correct.
SupCon also cares about the **geometry of the embedding space**:
- Same-class samples should cluster tightly together
- Different-class samples should be pushed apart

For spice classification, this is critical for hard-negative pairs (coriander vs cumin).
Even when both are predicted correctly, SupCon refines the embedding so they are
far apart — making the model more confident and robust.

### Forward Method

```python
def forward(self, features, labels):
    """
    features: (B, 128) — L2-normalized projection embeddings (from forward_contrastive)
    labels:   (B,)    — integer class labels 0–10
    """
```

### Step-by-Step Computation

**Step 1: Cosine similarity matrix**
```python
sim = torch.matmul(features, features.T) / self.temperature  # (B, B)
```
Since features are L2-normalized, `matmul(f, f.T)` computes cosine similarity.
Dividing by temperature `τ=0.07` sharpens the distribution (lower τ → sharper contrast).

**Step 2: Numerical stability**
```python
sim_max, _ = sim.max(dim=1, keepdim=True)
sim = sim - sim_max.detach()    # subtract row max before exp()
```
Prevents `exp(sim)` from overflowing. Same trick as logsumexp.

**Step 3: Build positive mask**
```python
labels_col = labels.view(-1, 1)
pos_mask = (labels_col == labels_col.T).float()   # 1 where same class
pos_mask = pos_mask - eye    # exclude self-pairs (diagonal)
```
A "positive pair" = two different samples from the same class within the same batch.
Example: if batch has 3 "cumin" images, each cumin is a positive for the other two.

**Step 4: Compute log-probabilities**
```python
exp_sim = torch.exp(sim) * (1 - eye)   # exp similarities, no self
log_prob = sim - log(exp_sim.sum(dim=1, keepdim=True))
```
This is log(exp(sim_ij) / sum_k(exp(sim_ik))) — the log-probability that i "matches" j
in a contrastive sense.

**Step 5: Average over positives**
```python
n_pos = pos_mask.sum(dim=1)
mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
loss = -mean_log_prob_pos    # maximize log-prob of positives = minimize this
```
For each anchor, average the log-probability over all its positive pairs.
Negate to make it a loss (minimize = maximize positive similarity).

**Step 6: Handle anchors with no positives**
```python
valid = n_pos > 0
if valid.sum() == 0:
    return torch.tensor(0.0, requires_grad=True)
return loss[valid].mean()
```
If a batch has only 1 sample per class (no positive pairs), SupCon loss is 0.
This can happen with small batch sizes or many classes.

### Temperature Parameter `τ = 0.07`

Lower temperature → harder contrast (sharper separation).
`0.07` is the standard value from the original paper.

| τ value | Effect |
|---|---|
| High (1.0+) | Soft, all pairs contribute similarly |
| Medium (0.1–0.5) | Moderate separation |
| Low (0.07) | Hard contrast — only very similar pairs matter |

---

## Class: `CombinedLoss`

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, label_smoothing=0.1):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.supcon = SupConLoss()

    def forward(self, logits, proj_feats, labels):
        return 0.5 * CE(logits, labels) + 0.5 * SupCon(proj_feats, labels)
```

**Used in Phase 3 only.**

Phase 3 needs both:
- **CE loss** — directly optimizes classification accuracy (via `logits`)
- **SupCon loss** — maintains embedding space geometry (via `proj_feats`)

`alpha=0.5` balances both equally. This was tuned empirically.

### Label Smoothing in CE

```python
CrossEntropyLoss(label_smoothing=0.1)
```

Instead of hard targets (0 or 1), label smoothing uses:
- Correct class: `1 - 0.1 = 0.9`
- Other classes: `0.1 / (11 - 1) = 0.01` each

This prevents the model from being overconfident and improves generalization.

### Why not use CombinedLoss in Phase 1?

Phase 1 uses only CE (no SupCon). At the start of training, embeddings are random
so SupCon provides no useful gradient signal. Introducing it only in Phase 2
(pure SupCon) and Phase 3 (combined) follows curriculum learning principles.
