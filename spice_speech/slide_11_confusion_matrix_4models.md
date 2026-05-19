# SLIDE 11 — CONFUSION MATRIX: FUSION, EFFICIENTNET, RESNET, VIT
# Speaker: Ramim (continues) | Time: ~1:00

This slide shows the confusion matrices for our four neural network models.

*(point to the slide — 4 confusion matrices)*

A confusion matrix shows — for every spice class —
how often the model predicted it correctly,
and when it made mistakes, which class did it confuse it with.

The **diagonal** of each matrix — the dark blue squares going from top-left to bottom-right —
represents correct predictions.
A perfect model would have a fully dark diagonal and white everywhere else.

**Let's look at our Fusion model** — top left.
The diagonal is almost completely dark blue.
The numbers off the diagonal — the mistakes — are extremely small.
This visually confirms our 99.68% accuracy.

**EfficientNet-B4 image only** — top right.
Also a strong diagonal, but you can notice slightly more off-diagonal errors —
especially for coriander, cumin, and turmeric.
This is exactly where the texture and color branches help.

**ResNet-50** — bottom left.
Still strong overall, but slightly more spread in certain rows —
showing ResNet struggles more with the hard-negative pairs.

**ViT-Base** — bottom right.
Similar to EfficientNet — very clean diagonal,
with only minor errors in the challenging pairs.

The key takeaway from these matrices:
The hardest pairs — coriander vs cumin, paprika vs turmeric —
are where our fusion model shows the most improvement
compared to image-only models.
Multi-modal fusion directly solves the hard cases.

---
💡 TIP: You don't need to read every number — just point to the diagonal and say "dark = correct."
       Focus on the differences between the 4 matrices.
       This slide is visual — let the audience look at it while you talk.
