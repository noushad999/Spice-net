# SLIDE 08 — METHODOLOGY: END-TO-END PIPELINE DIAGRAM
# Speaker: Rafi (continues) | Time: ~0:45

This diagram shows the complete end-to-end pipeline of SpiceNet —
from raw data to final prediction.

*(point to the pipeline diagram)*

**Stage 1 — Dataset:**
We start with the SpiceSpectrum Dataset —
11 spice classes, approximately 11,000 images, perfectly class-balanced.

**Stage 2 — Preprocessing:**
Images are split 70/10/20 for train, validation, and test.
Data augmentation is applied — flipping, rotation, color jitter, and noise.
Optionally, SAM background removal is applied for the ablation variant.

**Stage 3 — Feature Extraction:**
Two parallel paths:
The top path takes 224×224 images through EfficientNet-B4 — giving 1792-d CNN features.
The bottom path takes 512×512 images through LBP+GLCM — giving 58-d texture —
and HSV Histogram — giving 100-d color features.

**Stage 4 — Three-Phase Training:**
Phase 1 pre-trains the backbone.
Phase 2 applies contrastive fine-tuning.
Phase 3 trains full fusion end-to-end.
Each phase saves a checkpoint and passes it to the next.

**Stage 5 — Evaluation:**
The final model is evaluated on the held-out test set.
Results: Top-1 Accuracy 99.68%, Top-5 Accuracy 99.95%,
Inference Time 2.70 milliseconds per image.

This is the complete story of how a raw spice image
becomes a correct prediction in under 3 milliseconds.

---
💡 TIP: Move your hand across the diagram left to right as you explain each stage.
       This is a visual slide — let the diagram do the talking, you just narrate.
       Keep this section brief — 45 seconds maximum.
