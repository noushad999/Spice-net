# SLIDE 09 — IMPLEMENTATION: TOOLS AND FRAMEWORKS
# Speaker: Samira | Time: ~0:45

Thank you, Rafi.

Let me quickly walk you through the tools and frameworks we used to build this system.

*(point to the table on slide)*

**Language:** We used Python 3.10 — the standard language for deep learning research today.

**Deep Learning Framework:** PyTorch — industry-standard for building and training neural networks.
We also used the timm library — which gave us access to pre-trained EfficientNet-B4
with a single line of code.

**Classical Features:** For LBP and GLCM texture extraction, we used scikit-image.
For HSV histogram and image processing, we used OpenCV.

**Augmentation:** We used Albumentations —
a fast, flexible augmentation library that is significantly faster than torchvision transforms.

**Interpretability:** We implemented Grad-CAM
to visualize where exactly the model is looking in each image.
This helped us verify that the model focuses on the spice — not the background.

**Experiment Tracking:** We used Weights and Biases — W&B —
to track training curves, losses, and accuracy across all experiments.

**Model Sharing:** The trained model is available on HuggingFace Hub
for the research community to use and build upon.

Together, these tools allowed us to go from raw images to a fully trained,
evaluated, and deployable model — with clean, reproducible code.

---
💡 TIP: This is a quick slide — spend no more than 45 seconds.
       Just name each tool and one line about why you chose it.
       Don't go into technical depth here.
