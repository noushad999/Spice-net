# SLIDE 12 — CONFUSION MATRIX: SVM AND IMAGE-ONLY
# Speaker: Ramim (continues) | Time: ~0:45

This slide shows two more confusion matrices —
for our weakest and intermediate models.

**Top — SVM with HOG and Color Histogram:**

Look at this matrix. It is a mess of numbers everywhere.
There is no clear dark diagonal — predictions are scattered across all classes.
This confirms the 32.49% accuracy we saw in the baseline table.

The SVM simply cannot distinguish between visually similar spices.
Coriander gets confused with cumin, ginger, nutmeg —
almost every class gets confused with almost every other class.

This dramatically shows why traditional machine learning is not enough
for fine-grained spice classification.

**Bottom — Image-Only Model (Phase 1 only):**

Now look at the bottom matrix — our Phase 1 image-only model.
The diagonal is much clearer — most predictions are correct.
But compare it to the fusion matrix on the previous slide —
there are slightly more errors, especially in rows for cumin, coriander,
and the powder spices like turmeric and paprika.

This proves that adding texture and color branches in Phase 3
directly reduces those specific errors.

The visual contrast between SVM (32%) and our fusion model (99.68%)
tells the entire story of why deep learning with multi-modal fusion
is the right approach for this problem.

---
💡 TIP: When showing the SVM matrix, shake your head slightly or say "not good" —
       it helps emphasize the contrast.
       Then say "now look at this one" pointing to the image-only matrix
       to show the progression.
