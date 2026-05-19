# SLIDE 07 — METHODOLOGY: FUSION + 3-PHASE TRAINING
# Speaker: Rafi (continues) | Time: ~1:15

Now — after the three branches produce their outputs,
how do we combine them intelligently?

**AttentionFusion Module:**

The three branch outputs — 1792, 256, and 128 dimensions —
are concatenated into a **2176-dimensional** vector.

Then, a small neural network learns three importance weights —
one for each branch — using a softmax gate.
These weights sum to exactly 1.0, so they are interpretable as percentages.

The model learns: for a yellow powder like turmeric, weight the color branch more.
For a textured seed like coriander, weight the texture branch more.
This weighting happens **automatically during training** — no manual tuning needed.

The final weighted vector goes into the **fusion_head** —
a classifier MLP from 2176 to 512 to 11 classes.

The total model size is **21.6 million parameters** —
with EfficientNet-B4 alone contributing 19.3 million.

**Now — the Three-Phase Training Strategy:**

*(point to the training table on slide)*

We do NOT train everything at once. We use curriculum learning — training in stages.

**Phase 1 — Supervised Pre-training — 30 epochs:**
We train EfficientNet-B4 and the image head only.
Loss: Cross-Entropy with Label Smoothing.
The model learns basic spice classification from images alone.

**Phase 2 — Contrastive Fine-tuning — 10 epochs:**
We apply Supervised Contrastive Loss — SupCon Loss.
This trains the backbone and projection head
specifically to push hard-negative pairs apart —
cumin away from coriander, turmeric away from mustard.
Only the backbone and projection head are trained here.
All other parts are frozen.

**Phase 3 — Full Fusion Training — 10 epochs:**
Now ALL branches are active — CNN, texture, and color.
Loss: Combined CE plus SupCon, weighted equally at 0.5 each.
The AttentionFusion module and fusion_head are trained.
This is the final, complete model.

This step-by-step approach is why our model is so much stronger
than training everything together from scratch.

---
💡 TIP: Emphasize "automatically during training" — it shows the model's intelligence.
       Say Phase 1, Phase 2, Phase 3 with a clear pause between each.
       The training table on slide is your visual aid — point to each row.
