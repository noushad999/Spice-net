"""Save baseline comparison results and generate bar chart."""
import sys, os
_base = "/mnt/d/SpiceNet" if os.path.exists("/mnt/d/SpiceNet") else "D:/SpiceNet"
sys.path.insert(0, _base)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results = [
    {"model": "SVM (HOG + Color)",         "top1": 0.3249, "top5": None,   "f1_macro": 0.3069},
    {"model": "ResNet-50 (fine-tuned)",     "top1": 0.9936, "top5": 0.9995, "f1_macro": 0.9936},
    {"model": "EfficientNet-B4 (image-only)","top1": 0.9959, "top5": 1.0000, "f1_macro": 0.9959},
    {"model": "ViT-Base/16 (fine-tuned)",   "top1": 0.9973, "top5": 0.9995, "f1_macro": 0.9973},
    {"model": "SpiceFusionNet (ours)",      "top1": 0.9968, "top5": 1.0000, "f1_macro": 0.9968},
]

out = Path("D:/SpiceNet/outputs")
out.mkdir(parents=True, exist_ok=True)

with open(out / "baseline_comparison.json", "w") as f:
    json.dump({"baselines": results}, f, indent=2)
print("Saved baseline_comparison.json")

# --- Bar chart ---
models    = [r["model"] for r in results]
top1_vals = [r["top1"] * 100 for r in results]
f1_vals   = [r["f1_macro"] * 100 for r in results]

x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, top1_vals, width, label="Top-1 Accuracy (%)", color="#4C72B0", zorder=3)
bars2 = ax.bar(x + width/2, f1_vals,   width, label="Macro F1-Score (%)",  color="#DD8452", zorder=3)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Highlight our model
ax.get_xticklabels()
bar_colors = ["#4C72B0"] * len(models)
bar_colors[-1] = "#2ca02c"
for i, bar in enumerate(bars1):
    if i == len(models) - 1:
        bar.set_color("#2ca02c")
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)
for i, bar in enumerate(bars2):
    if i == len(models) - 1:
        bar.set_color("#98df8a")
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Baseline Comparison on SpiceSpectrum Test Set", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, 110)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(out / "baseline_comparison.png", dpi=150)
plt.close()
print("Saved baseline_comparison.png")
