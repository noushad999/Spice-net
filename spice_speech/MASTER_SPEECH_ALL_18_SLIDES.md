# ═══════════════════════════════════════════════════════════════
#  SPICENET — COMPLETE 10-MINUTE SPEECH  |  ALL 18 SLIDES
#  CSE 414 — Machine Learning and Deep Learning Lab
#  University of Asia Pacific
#  Presented to: Shahiar Raj Sir
# ═══════════════════════════════════════════════════════════════
#
#  SPEAKERS:
#  Ramim  = Md. Noushad Jahan Ramim  (22201257)
#  Maisha = Maisha Sameha            (22201266)
#  Samira = Samira Islam             (22201262)
#  Rafi   = Junaid Abedin Rafi       (22201265)
#
#  TIMING BREAKDOWN:
#  Slide 01  Title              Ramim   0:00 – 0:45
#  Slide 02  Contents           Ramim   0:45 – 1:15
#  Slide 03  Objective          Ramim   1:15 – 2:30
#  Slide 04  Societal Impact    Maisha  2:30 – 3:30
#  Slide 05  Dataset            Samira  3:30 – 4:45
#  Slide 06  Methodology-1      Rafi    4:45 – 6:00
#  Slide 07  Methodology-2      Rafi    6:00 – 7:15
#  Slide 08  Pipeline Diagram   Rafi    7:15 – 8:00
#  Slide 09  Implementation     Samira  8:00 – 8:45
#  Slide 10  Results Baseline   Ramim   8:45 – 10:15
#  Slide 11  CM - 4 Models      Ramim   10:15 – 11:15
#  Slide 12  CM - SVM+Image     Ramim   11:15 – 12:00
#  Slide 13  GradCAM            Rafi    12:00 – 13:00
#  Slide 14  Training Curves    Rafi    13:00 – 14:00
#  Slide 15  Ablation Findings  Samira  14:00 – 15:00
#  Slide 16  Team Contributions Maisha  15:00 – 15:45
#  Slide 17  Conclusion         Maisha  15:45 – 16:45
#  Slide 18  References + TY    All     16:45 – 17:15
#  Q&A Buffer                   All     17:15 – 20:00
#
# ═══════════════════════════════════════════════════════════════


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 01 — TITLE  |  RAMIM  |  0:00 – 0:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Good morning / Good afternoon, everyone.

Let me start with a simple question —
Can you tell the difference between cumin and coriander, just by looking at them?

*(pause 2 seconds)*

Most people cannot. Even food experts make mistakes.
Now imagine a computer doing that — correctly — thousands of times per second.
That is exactly what we built.

My name is Md. Noushad Jahan Ramim.
Together with Maisha Sameha, Samira Islam, and Junaid Abedin Rafi —
we present our project for CSE 414:

"Deep Convolutional Neural Networks for Fine-Grained Spice Image Classification."

Presented to our respected teacher, Shahiar Raj sir.

In the next 10 minutes — the problem, the solution, and the results.
Let's begin.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 02 — CONTENTS  |  RAMIM  |  0:45 – 1:15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Here is what we will cover today.

Project Objective — the problem and our goal.
Societal Impact — why this matters to the real world.
Dataset — what data we used and how we prepared it.
Methodology — how our model is designed and trained.
Implementation — the tools we built with.
Results and Analysis — numbers, confusion matrices, Grad-CAM, training curves.
Ablation Findings — proof that every design choice was the right one.
Team Contributions, CEP Mapping, Conclusion, and References.

Let's start with the Project Objective.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 03 — PROJECT OBJECTIVE  |  RAMIM  |  1:15 – 2:30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

So — what is the problem?

Spices are visually nearly identical.
Cumin versus caraway — both tiny brown seeds.
Turmeric versus mustard powder — both yellow powders.
Even trained food industry workers get confused.

And existing computer systems?
They either ignore spices completely —
or label everything as "seasoning" — which is useless.

This leads to mislabeling, quality failures, and health risks in the food industry.

Our goal:
Build SpiceFusionNet —
a multi-modal deep learning model that classifies 11 spice types
by fusing image, texture, and color features.

Accurate enough for real-world deployment —
in smart kitchen assistants, food quality control, and retail systems.

Our 11 classes:
Black Pepper, Cardamom, Cinnamon, Cloves, Coriander,
Cumin, Ginger, Nutmeg, Paprika, Saffron, Turmeric.

Eleven spices. Eleven challenges. One unified model.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 04 — SOCIETAL IMPACT  |  MAISHA  |  2:30 – 3:30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Ramim.

Why does this matter beyond a university project?

First — Food Safety.
Spice adulteration is a global problem.
Mislabeled spices enter food supply chains every day.
Our system catches fraud before it reaches consumers.

Second — Smart Kitchens.
Imagine your phone camera identifying any spice instantly — no label needed.
Our model makes that possible.

Third — Retail and E-Commerce.
Automated batch verification saves companies significant time and money.

Who benefits?
Consumers get safer food.
Food companies get automated quality control.
Retailers get faster inventory management.
Researchers get our SpiceSpectrum dataset.

What makes us unique?
Purpose-built for spices — custom dataset, spice-specific augmentation,
contrastive training for hard pairs.

This is a real solution to a real problem.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 05 — DATASET  |  SAMIRA  |  3:30 – 4:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Maisha.

No good spice dataset existed — so we built our own.

SpiceSpectrum Dataset:
11,000 images across 11 classes.
1,000 images per class — perfectly balanced.
Sources: Open Images, iNaturalist, and self-captured photos.
Different lighting, backgrounds, whole and ground forms.

Data Split:
70% Training — 10% Validation — 20% Testing.
Test set is completely held out — results are honest and trustworthy.

Preprocessing:
Resize to 512×512 for feature extraction, 224×224 for CNN.
Normalize with ImageNet mean and standard deviation.

Augmentation:
Horizontal and vertical flips, rotation up to 30 degrees,
color jitter, Gaussian blur, Gaussian noise, coarse dropout.
All designed to simulate real-world variation.

We also created a SAM variant using Meta's Segment Anything Model —
used in our ablation study — which we discuss later.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 06 — METHODOLOGY: THREE BRANCHES  |  RAFI  |  4:45 – 6:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Samira.

SpiceFusionNet has three parallel branches — each capturing different information.

Branch 1 — CNN Backbone:
Input: 224×224×3 raw image.
Architecture: EfficientNet-B4, pre-trained on ImageNet, with global average pooling.
Output: 1792-dimensional feature vector.
Captures deep visual patterns — shapes, edges, overall appearance.

Branch 2 — Texture Branch:
Input: LBP (10-d) plus GLCM (48-d) — total 58 dimensions.
Architecture: MLP — 58 to 128 to 256 — BatchNorm, ReLU, Dropout (0.3).
Output: 256-dimensional texture embedding.
Separates coriander from cumin based on surface texture differences.

Branch 3 — Color Branch:
Input: HSV Histogram — H:36, S:32, V:32 — total 100 dimensions.
Architecture: MLP — 100 to 64 to 128 — BatchNorm, ReLU, Dropout (0.3).
Output: 128-dimensional color embedding.
Separates turmeric yellow from paprika orange with precision.

Three branches. Three information types. Each captures what the others cannot.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 07 — METHODOLOGY: FUSION + TRAINING  |  RAFI  |  6:00 – 7:15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AttentionFusion Module:
Three branch outputs — 1792 plus 256 plus 128 — concatenated to 2176 dimensions.
A softmax gate learns importance weights: a_img, a_tex, a_col — summing to 1.
The weighted fused vector goes to fusion_head — MLP from 2176 to 512 to 11 classes.
Total model size: 21.6 million parameters.

Three-Phase Training Strategy:

Phase 1 — Supervised Pre-training — 30 epochs:
Trains EfficientNet-B4 and image head only.
Loss: Cross-Entropy with Label Smoothing.
Learns basic spice recognition.

Phase 2 — Contrastive Fine-tuning — 10 epochs:
Trains backbone and projection head only — all else frozen.
Loss: SupCon Loss.
Pushes hard-negative pairs apart in embedding space —
coriander away from cumin, turmeric away from mustard.

Phase 3 — Full Fusion Training — 10 epochs:
All branches active — CNN, texture, color.
Loss: Combined CE plus SupCon, weighted 0.5 each.
Trains AttentionFusion and fusion_head.

Curriculum learning — each phase builds on the previous one.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 08 — PIPELINE DIAGRAM  |  RAFI  |  7:15 – 8:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This diagram shows the complete end-to-end pipeline.

Stage 1 — Dataset: SpiceSpectrum, 11 classes, 11,000 images, class-balanced.

Stage 2 — Preprocessing: 70/10/20 split, augmentation, optional SAM removal.

Stage 3 — Feature Extraction:
Top path — 224×224 through EfficientNet-B4 — 1792-d CNN features.
Bottom path — 512×512 through LBP+GLCM — 58-d texture,
and HSV Histogram — 100-d color.

Stage 4 — Three-Phase Training: Phase 1, Phase 2, Phase 3 — each saving a checkpoint.

Stage 5 — Evaluation:
Top-1 Accuracy 99.68%. Top-5 Accuracy 99.95%. Inference Time 2.70 milliseconds.

This is how a raw spice image becomes a correct prediction in under 3 milliseconds.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 09 — IMPLEMENTATION  |  SAMIRA  |  8:00 – 8:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Rafi.

Here are the tools we used.

Language: Python 3.10.
Deep Learning: PyTorch, with the timm library for pre-trained EfficientNet-B4.
Classical Features: scikit-image for LBP and GLCM. OpenCV for HSV histograms.
Augmentation: Albumentations — significantly faster than torchvision.
Interpretability: Grad-CAM — to visualize what the model focuses on.
Experiment Tracking: Weights and Biases — W&B — for all training curves and metrics.
Model Sharing: HuggingFace Hub — for community access to our trained model.

These tools allowed clean, reproducible, production-quality code.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 10 — RESULTS: BASELINE COMPARISON  |  RAMIM  |  8:45 – 10:15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Samira. Now — results.

Evaluation metrics: Top-1 and Top-5 Accuracy, Macro F1, Per-class Precision and Recall,
Confusion Matrix, and Inference Time.

Baseline Comparison:

SVM with HOG and Color Histogram — Top-1: 32.49%, F1: 30.69%.
Traditional machine learning simply cannot handle this problem.

ResNet-50 fine-tuned — Top-1: 99.36%.
Strong — but an older architecture.

EfficientNet-B4 image only — Top-1: 99.59%.
Our own backbone — but without fusion. Already very strong.

ViT-Base fine-tuned — Top-1: 99.73%.
Top performer in accuracy — but at 4.3 milliseconds per image.

SpiceFusionNet — our model — Top-1: 99.68%, F1: 99.68%.

The accuracy gap between us and ViT is only 0.05 percentage points — essentially equal.
But our inference time is 2.70 milliseconds — versus ViT's 4.3 milliseconds.
We are 37% faster while delivering nearly identical accuracy.

For real-world deployment, speed is not optional. Our model wins where it matters most.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 11 — CONFUSION MATRIX: FUSION, EFFNET, RESNET, VIT  |  RAMIM  |  10:15 – 11:15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Confusion matrices — showing how each model performs class by class.

The dark blue diagonal means correct predictions.
A perfect model has a fully dark diagonal with nothing else.

Our Fusion model — top left.
The diagonal is almost completely dark. Errors are minimal.
Visually confirms 99.68% accuracy.

EfficientNet-B4 image only — top right.
Strong diagonal, but slightly more off-diagonal errors —
especially in coriander, cumin, and the powder spices.
This is exactly where adding texture and color branches helps.

ResNet-50 — bottom left.
Good overall, but more spread — ResNet struggles more with hard pairs.

ViT-Base — bottom right.
Clean diagonal — competitive with our model in accuracy.

Key message:
Hard-negative pairs — coriander vs cumin, paprika vs turmeric —
are where our fusion model shows the most improvement over image-only models.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 12 — CONFUSION MATRIX: SVM AND IMAGE  |  RAMIM  |  11:15 – 12:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two more matrices — for comparison.

SVM matrix — top.
No clear diagonal. Predictions are scattered everywhere.
32.49% accuracy — visually confirmed.
Traditional ML cannot handle this problem.

Image-only model — bottom.
Clear diagonal — Phase 1 EfficientNet is already strong.
But compare it to the fusion matrix on the previous slide —
slightly more errors in cumin, coriander, and powder spices.

Adding texture and color branches in Phase 3 directly reduces those errors.

The visual story: SVM at 32% versus our fusion model at 99.68% —
shows exactly why deep learning with multi-modal fusion is the right approach.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 13 — GRADCAM VISUALIZATION  |  RAFI  |  12:00 – 13:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Ramim.

Where exactly is our model looking?

We used Grad-CAM — Gradient-weighted Class Activation Maps —
to visualize which parts of each image most influenced the model's decision.

Red and hot colors: high attention. Blue and cool colors: ignored.

Look at correctly predicted examples.
The heatmap focuses on the spice itself —
texture of seeds, shape of leaves, color of powders.
Background — table, bowl, hand — mostly ignored.

This means our model learned to look at the right things —
without us explicitly telling it where to look.

Look at incorrect predictions — labeled in red.
The heatmap shows attention going to ambiguous or background regions.
These are genuinely hard cases — and Grad-CAM shows us exactly why.

Grad-CAM gives us interpretability.
We can trust predictions because we can see the reasoning.
This is critical for food safety applications.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 14 — TRAINING CURVES  |  RAFI  |  13:00 – 14:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1 Training Curves — 30 epochs:

Loss curve: starts above 2.5, drops quickly within the first 5 epochs,
stabilizes by epoch 30. Training and validation stay close — no overfitting.

Accuracy curve: jumps from near zero to above 95% in just a few epochs,
then gradually reaches 99%.

Learning rate: warm-up, then cosine annealing — smooth, controlled decrease.

Phase 3 Training Curves — 10 epochs:

Phase 3 starts at much higher accuracy than Phase 1 did —
because we initialize from Phase 2 weights, not from scratch.

Loss drops quickly in the first 2 epochs, then stabilizes.
Validation accuracy reaches above 99% and holds steady.
Learning rate is very small — 1e-5 — careful fine-tuning.

These curves confirm our curriculum training works exactly as designed.
Each phase builds cleanly on the previous one.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 15 — KEY ABLATION FINDINGS  |  SAMIRA  |  14:00 – 15:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Rafi.

Ablation studies — proving every design decision was correct.

A1 — Component Contribution:
Image only, plus texture, plus color, full fusion.
Each addition improves accuracy. Full fusion is always best.
Every branch contributes uniquely.

A2 — Contrastive Loss — Phase 2:
Without Phase 2 SupCon — accuracy drops significantly.
Contrastive training is critical for hard-negative pairs.
Coriander vs cumin. Paprika vs turmeric. Phase 2 solves them.

A3 — Augmentation Policy:
No augmentation: around 93%.
Basic augmentation: around 96.5%.
Our spice-specific full policy: 99.68%.
Domain-specific augmentation is essential — not optional.

A4 — Backbone Selection:
Tested MobileNetV3, EfficientNet B0, B2, B4, ResNet-50.
EfficientNet-B4 gives best accuracy-to-speed tradeoff. Correct choice confirmed.

A5 — Background Removal with SAM:
Removing backgrounds did NOT help — slightly hurt performance.
Grad-CAM confirmed: the model already focuses on the spice naturally.
Background gives useful context sometimes.

Every design decision validated. Every choice was the right one.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 16 — TEAM CONTRIBUTIONS  |  MAISHA  |  15:00 – 15:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you, Samira.

Ramim — model architecture design, EfficientNet-B4 integration,
Phase 1 and Phase 2 training, report writing.

Maisha — that's me — dataset preprocessing, augmentation pipeline,
SAM variant creation, report writing.

Samira — texture and color branch implementation,
AttentionFusion module, all ablation studies, report writing.

Rafi — baseline model evaluation, Grad-CAM visualization,
results analysis, presentation preparation.

Together we covered every aspect of this project — data to model to evaluation.
We are proud of what we built together.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 17 — CONCLUSION  |  MAISHA  |  15:45 – 16:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let me bring everything together.

What we built:
SpiceFusionNet — a multi-modal CNN fusing image, texture, and color
to classify 11 spice types with high accuracy.

What we proved:
Contrastive fine-tuning on hard-negative pairs significantly reduced confusion.
The system is ready for real-time deployment in smart kitchen,
food quality control, and retail applications.

Challenges we overcame:
High visual similarity required our multi-modal approach.
No existing dataset — we built our own.
Background clutter — addressed with SAM segmentation.

Future Work:
Scale to 30+ spice categories.
Deploy as mobile app using lightweight MobileNetV3.
Extend to multi-spice blend recognition and adulteration detection.

We started with a hard problem.
We built a smart solution.
We proved it works.

SpiceFusionNet is not the end of this journey — it is the beginning.

Thank you so much for your time.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 18 — REFERENCES + THANK YOU  |  RAMIM  |  16:45 – 17:15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Our key references:

EfficientNet — Tan and Le, ICML 2019.
Supervised Contrastive Learning — Khosla et al., NeurIPS 2020.
Grad-CAM — Selvaraju et al., ICCV 2017.
Segment Anything — Kirillov et al., ICCV 2023.
SpiceSpectrum Dataset — Data in Brief, Elsevier, 2025.

We stand on the shoulders of great researchers.
We hope our work adds one more step forward.

*(All together)*

Thank you, Shahiar Raj sir.
Thank you everyone for listening.

We are happy to take any questions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q&A QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Why EfficientNet-B4 specifically?
A: We tested 5 backbones. B4 gave best accuracy-speed balance. (see A4 ablation)

Q: Why does your model score 99.68% but ViT scores 99.73%?
A: Our model is 37% faster — 2.70ms vs 4.30ms. For real-world use, speed matters.

Q: What is SupCon loss in simple terms?
A: Normal training gets the right answer. SupCon also makes similar spices cluster
   together in the model's memory — and pushes confusing pairs far apart.

Q: Why didn't background removal help?
A: Grad-CAM showed the model already focuses on the spice, not the background.
   Sometimes background context actually helps distinguish whole vs ground spice.

Q: How long did training take?
A: 50 total epochs. Approximately 5-10 hours on GPU. Inference: 2.70ms per image.

Q: Can this work on a mobile phone?
A: Not yet — 21.6M parameters is too large. Future work: compress with MobileNetV3.
