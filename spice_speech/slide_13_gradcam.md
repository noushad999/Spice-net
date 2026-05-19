# SLIDE 13 — GRADCAM VISUALIZATION
# Speaker: Rafi | Time: ~1:00

Thank you, Ramim.

This slide answers a very important question:
**Where exactly is our model looking when it makes a prediction?**

We used **Grad-CAM — Gradient-weighted Class Activation Maps** —
a technique that highlights which regions of the image
most influenced the model's decision.

*(point to the grid of images)*

Each pair of images shows the original photo on the left,
and the Grad-CAM heatmap overlay on the right.
The **red and hot colors** show where the model is focused most.
The **blue and cool colors** show areas the model ignored.

Look at the correctly predicted examples —
the heatmap consistently focuses on the spice itself —
the texture of the seed, the shape of the leaf, the color of the powder.
The background — the table, the bowl, the hand — is mostly ignored.

This is very important. It tells us our model has learned
to look at the right things — without us explicitly telling it where to look.

Now look at some of the incorrect predictions — labeled in red at the top.
For example — a saffron image predicted as cardamom.
The heatmap shows the model focused on the wrong region —
confirming that these are genuinely hard cases
where even the model's attention goes to ambiguous features.

Grad-CAM gives us interpretability —
we can trust the model's predictions because we can see its reasoning.
This is critical for any real-world food safety application.

---
💡 TIP: Point to specific image pairs as you explain — red title = wrong prediction.
       The audience finds this slide visually interesting — give them a moment to look.
       The key message: "model looks at the spice, not the background."
