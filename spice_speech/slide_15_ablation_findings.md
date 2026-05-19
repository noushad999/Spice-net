# SLIDE 15 — KEY ABLATION FINDINGS
# Speaker: Samira | Time: ~1:00

Thank you, Rafi.

Now — ablation studies. This is where we prove that every design decision we made
was the right one.

An ablation study means: we take one part of our system,
remove it or change it, and see what happens to performance.
If performance drops — that part was important.

We ran 5 ablation experiments.

**A1 — Component Contribution:**
We tested image-only, then added texture, then added color, then full fusion.
Each addition improved accuracy.
Full fusion is always the best.
This proves every branch contributes uniquely.

**A2 — Contrastive Loss — Phase 2:**
We compared training with and without Phase 2 SupCon fine-tuning.
Without Phase 2 — accuracy drops significantly.
Phase 2 is critical specifically for the hard-negative pairs —
coriander vs cumin, paprika vs turmeric.
Without contrastive training, the model confuses these pairs more often.

**A3 — Augmentation Policy:**
No augmentation: accuracy drops to around 93%.
Basic augmentation: around 96.5%.
Our spice-specific full augmentation policy: 99.68%.
Domain-specific augmentation is not optional — it is essential.

**A4 — Backbone Selection:**
We tested MobileNetV3, EfficientNet B0, B2, B4, and ResNet-50.
EfficientNet-B4 gave the best accuracy-to-speed tradeoff.
This confirms our backbone choice was optimal.

**A5 — Background Removal with SAM:**
Removing backgrounds did NOT improve accuracy — it actually slightly hurt performance.
Our Grad-CAM visualizations confirmed the reason:
the model already focuses on the spice region naturally.
Background removal sometimes removes useful context.

Every design decision is validated. Every choice was correct.

---
💡 TIP: Say "Every design decision is validated" at the end with confidence.
       For each finding, say the result clearly — don't be vague.
       This slide shows scientific rigor — sound professional and precise.
