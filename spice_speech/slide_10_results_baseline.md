# SLIDE 10 — RESULTS AND ANALYSIS: BASELINE COMPARISON
# Speaker: Ramim | Time: ~1:30

Thank you, Samira.

Now — the moment everyone has been waiting for. Results.

Let me first explain how we measured our model's performance.

**We used five evaluation metrics:**
Top-1 and Top-5 Classification Accuracy,
Macro-averaged F1 Score,
Per-class Precision and Recall,
Confusion Matrix focusing on hard-negative pairs,
and Inference time in milliseconds per image.

These metrics together give a complete and honest picture of the model's quality.

**Baseline Comparison:**

*(point to the table)*

We compared SpiceFusionNet against four strong baselines on the exact same dataset.

First — **SVM with HOG and Color Histogram** — the classical machine learning approach.
Top-1 Accuracy: **32.49%**. F1 Score: **30.69%**.
Traditional ML simply cannot handle the complexity of fine-grained spice images.

Second — **ResNet-50**, fine-tuned on our dataset.
Top-1 Accuracy: **99.36%**. A strong result — but ResNet is a much older architecture.

Third — **EfficientNet-B4 image only** — the same backbone we use, but without fusion.
Top-1 Accuracy: **99.59%**. 
Notice — this is already very high. But fusion makes it better.

Fourth — **ViT-Base/16**, the Vision Transformer, fine-tuned.
Top-1 Accuracy: **99.73%**.

And finally — **SpiceFusionNet, our model.**
Top-1 Accuracy: **99.68%**. F1 Score: **99.68%**.

Now — ViT-Base scores 99.73% and we score 99.68%.
The difference is only **0.05 percentage points** — essentially equal.

But here is the critical difference:
ViT-Base takes **4.3 milliseconds** per image.
SpiceFusionNet runs at **2.70 milliseconds** per image —
**37% faster** than ViT, while delivering nearly identical accuracy.

For real-world deployment, speed matters enormously.
Our model gives the best accuracy-to-speed tradeoff of all models tested.

---
💡 TIP: Don't skip the SVM result — "32.49%" is dramatic and shows how hard the problem is.
       The ViT comparison is important — explain WHY our model is still better (speed!).
       Say "37% faster" with confidence — it's a strong practical advantage.
