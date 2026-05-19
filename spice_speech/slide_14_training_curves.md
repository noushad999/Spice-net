# SLIDE 14 — TRAINING LOSS, ACCURACY AND LEARNING RATE CURVES
# Speaker: Rafi (continues) | Time: ~1:00

This slide shows how our model learned during training —
through loss curves, accuracy curves, and learning rate schedules.

*(point to P1 training curves — top row)*

**Phase 1 Training Curves — 30 epochs:**

Look at the Loss curve on the left.
Both training loss and validation loss start high — above 2.5 —
and drop quickly within the first 5 epochs.
By epoch 10, the model is already converging.
By epoch 30, both curves are flat and close together.
This means the model is NOT overfitting — training and validation behave similarly.

The Accuracy curve in the middle tells the same story —
accuracy jumps from near zero to above 95% in just a few epochs,
then gradually improves to around 99%.

The Learning Rate curve on the right shows our warm-up and cosine annealing schedule —
the rate starts very small, peaks, then smoothly decreases.
This helps the model find a good solution without getting stuck.

*(point to P3 training curves — bottom row)*

**Phase 3 Training Curves — 10 epochs:**

Notice — Phase 3 starts at a much higher accuracy than Phase 1 did.
That is because we initialize from Phase 2 weights — not from scratch.
The model is already strong when Phase 3 begins.

The loss drops quickly in the first 2 epochs,
then stabilizes at a low level.
Validation accuracy reaches above 99% and stays there.

The learning rate here is very small — 1e-5 — careful fine-tuning
to not disturb the good weights from Phase 1 and Phase 2.

These curves confirm that our 3-phase curriculum training
works exactly as designed — each phase builds on the previous one.

---
💡 TIP: Don't read numbers off the y-axis — just describe the shape of the curves.
       "Drops quickly, then flattens" — that's all you need to say.
       Point to train curve (blue) and val curve (orange) — show they stay close.
