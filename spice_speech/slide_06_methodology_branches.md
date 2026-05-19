# SLIDE 06 — METHODOLOGY: THREE BRANCH ARCHITECTURE
# Speaker: Rafi | Time: ~1:15

Thank you, Samira.

Now I will explain how SpiceFusionNet actually works — starting with its three branches.

*(point to the table on slide)*

When a human expert identifies a spice, they don't rely on just one thing.
They look at the overall appearance, feel the texture, and notice the color.
Our model does exactly the same — but using deep learning.

SpiceFusionNet has **three parallel branches**, each capturing different information.

**Branch 1 — CNN Backbone:**
Input: a raw image at 224×224×3 pixels.
Architecture: EfficientNet-B4, pre-trained on ImageNet, with global average pooling.
Output: a **1792-dimensional** feature vector.
This captures deep visual patterns — shapes, edges, and overall appearance.
Think of it as 1,792 different measurements about how the spice looks.

**Branch 2 — Texture Branch:**
Input: LBP features (10 dimensions) combined with GLCM features (48 dimensions),
giving a total of **58-dimensional** texture description.
Architecture: an MLP — 58 goes to 128, then to 256 — with BatchNorm, ReLU, and Dropout.
Output: **256-dimensional** texture embedding.
This is what separates coriander from cumin —
they look similar visually, but their surface texture tells a completely different story.

**Branch 3 — Color Branch:**
Input: HSV Histogram — H: 36 bins, S: 32 bins, V: 32 bins — total **100 dimensions**.
Architecture: MLP — 100 to 64 to 128 — with BatchNorm, ReLU, and Dropout.
Output: **128-dimensional** color embedding.
This separates turmeric yellow from paprika orange with high precision.

Three branches. Three types of information.
Each one capturing what the others cannot.

---
💡 TIP: Point to each row in the table as you explain each branch.
       Say the dimension numbers clearly — 1792, 256, 128.
       These numbers show you know your model deeply.
