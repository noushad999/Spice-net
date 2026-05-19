# SLIDE 05 — DATASET DESCRIPTION
# Speaker: Samira | Time: ~1:15

Thank you, Maisha.

Every powerful AI system starts with powerful data.
For spice classification, no clean, specialized dataset existed —
so we built our own.

We call it the **SpiceSpectrum Dataset.**

Here is what it contains:
**11,000 images** spread across 11 spice classes —
roughly **1,000 images per class**, making the dataset perfectly class-balanced.
No single spice is over-represented or under-represented.

We collected images from multiple sources —
Open Images, iNaturalist, and self-captured photos —
to ensure our model sees spices in all kinds of real-world conditions.
Different lighting. Different backgrounds.
Whole spices, ground spices, dried spices — all included.

**Data Split:**
We divided the data into three parts:
- **70% for Training** — the model learns from this
- **10% for Validation** — used during training to check performance
- **20% for Testing** — the final, unseen evaluation set

The test set is completely held out.
Our model never sees these images during training.
So our results are clean, honest, and trustworthy.

**Preprocessing:**
We resize images to 512×512 for feature extraction
and 224×224 for the CNN input.
We normalize pixel values using ImageNet statistics.

For augmentation, we apply:
horizontal and vertical flips, rotation up to 30 degrees,
color jitter, Gaussian blur, and Gaussian noise —
all designed to simulate real-world variation in spice photos.

We also created a variant using **SAM — Segment Anything Model** —
from Meta — to remove image backgrounds.
This variant was used in our ablation study, which we will discuss shortly.

---
💡 TIP: Pause after "So our results are clean, honest, and trustworthy."
       It's a strong line. Let it land.
