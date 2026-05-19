"""
SpiceFusionNet Architecture Figure — matches the WhatsApp reference style.
Output: outputs/arch_figure.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG    = "#FAFAFA"
C_CNN   = "#AED6F1"   # light blue
C_TEX   = "#FAD7A0"   # light orange
C_COL   = "#A9DFBF"   # light green
C_FUSE  = "#D2B4DE"   # light purple
C_HEAD  = "#F1948A"   # light red/salmon
C_OUT   = "#85C1E9"   # light blue output
C_EDGE  = "#5D6D7E"   # dark grey edges
C_TEXT  = "#1A252F"   # near black text
C_DASH  = "#7F8C8D"   # grey dashed box

FIG_W, FIG_H = 18, 10

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_facecolor(C_BG); fig.patch.set_facecolor(C_BG); ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, fc, ec=C_EDGE, lw=1.4, rad=0.35, zorder=4):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p); return p

def txt(cx, cy, s, fs=9, fw="normal", color=C_TEXT, ha="center", va="center",
        style="normal", zorder=5):
    ax.text(cx, cy, s, fontsize=fs, fontweight=fw, color=color,
            ha=ha, va=va, fontstyle=style, zorder=zorder)

def arrow(x0, y0, x1, y1, fc=C_EDGE, lw=1.6, ms=12):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=fc, lw=lw,
                                mutation_scale=ms,
                                connectionstyle="arc3,rad=0"),
                zorder=3)

def feat_bars(cx, cy, h, color, n=6, w=0.22):
    """Draw a column of small colored rectangles (feature vector visual)."""
    bar_h = h / (n + 1)
    for i in range(n):
        yy = cy - h/2 + bar_h*(i+1) - bar_h/2
        alpha = 0.3 + 0.7*(i/n)
        p = FancyBboxPatch((cx-w/2, yy-bar_h*0.4), w, bar_h*0.8,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor=color, edgecolor="white",
                           linewidth=0.5, alpha=alpha, zorder=4)
        ax.add_patch(p)

# ── Y positions for 3 branches ────────────────────────────────────────────────
Y_TOP = 7.8   # EfficientNet
Y_MID = 5.0   # Texture
Y_BOT = 2.3   # Color

# ── INPUT IMAGE ───────────────────────────────────────────────────────────────
# Spice-like image placeholder
rbox(1.3, 5.0, 1.8, 6.0, "#F0E6D3", ec="#C0A882", lw=1.5)
# mini color patches to simulate spice image
colors_spice = ["#C0392B","#E67E22","#F1C40F","#27AE60","#8E44AD","#2C3E50"]
for i, sc in enumerate(colors_spice):
    row, col2 = divmod(i, 2)
    px = 0.6 + col2*0.7; py = 6.8 - row*0.85
    rbox(px+0.3, py, 0.55, 0.55, sc, ec="white", lw=1, rad=0.08)
txt(1.3, 1.5, "Input Image", fs=9, fw="bold")
txt(1.3, 1.1, "(224 × 224 × 3)", fs=8, style="italic")

# ── Arrows from input to 3 branches ──────────────────────────────────────────
for yy in [Y_TOP, Y_MID, Y_BOT]:
    arrow(2.2, 5.0, 3.15, yy)

# ── BRANCH BOXES ─────────────────────────────────────────────────────────────
BX = 4.3   # branch box center x
BW, BH = 2.4, 1.4

# EfficientNet-B4
rbox(BX, Y_TOP, BW, BH, C_CNN)
txt(BX, Y_TOP+0.28, "EfficientNet-B4", fs=9.5, fw="bold")
txt(BX, Y_TOP-0.05, "(Pretrained)", fs=8.5)
txt(BX, Y_TOP-0.42, "Global Average Pooling", fs=7.8, style="italic")

# Texture Encoder
rbox(BX, Y_MID, BW, BH, C_TEX)
txt(BX, Y_MID+0.32, "Texture Encoder", fs=9.5, fw="bold")
txt(BX, Y_MID+0.0,  "(MLP)", fs=8.5)
txt(BX, Y_MID-0.32, "LBP + GLCM Features", fs=7.8, style="italic")

# Color Encoder
rbox(BX, Y_BOT, BW, BH, C_COL)
txt(BX, Y_BOT+0.32, "Color Encoder", fs=9.5, fw="bold")
txt(BX, Y_BOT+0.0,  "(MLP)", fs=8.5)
txt(BX, Y_BOT-0.32, "Color Histogram (HSV)", fs=7.8, style="italic")

# ── Feature vector bars ───────────────────────────────────────────────────────
FV_X = 6.1
feat_bars(FV_X, Y_TOP, 1.2, C_CNN,  n=7)
feat_bars(FV_X, Y_MID, 1.2, C_TEX,  n=7)
feat_bars(FV_X, Y_BOT, 1.2, C_COL,  n=7)

# Arrows branch → bars
for yy in [Y_TOP, Y_MID, Y_BOT]:
    arrow(BX+BW/2, yy, FV_X-0.15, yy)

# Dimension labels
txt(FV_X+0.45, Y_TOP, "1792-d", fs=8, fw="bold", color=C_EDGE)
txt(FV_X+0.45, Y_MID, "256-d",  fs=8, fw="bold", color=C_EDGE)
txt(FV_X+0.45, Y_BOT, "128-d",  fs=8, fw="bold", color=C_EDGE)

# ── ATTENTION FUSION ──────────────────────────────────────────────────────────
FU_X = 9.5; FU_Y = 5.0; FU_W = 2.8; FU_H = 5.2
rbox(FU_X, FU_Y, FU_W, FU_H, "#EBF5FB", ec=C_EDGE, lw=1.6)
txt(FU_X, 7.62, "AttentionFusion", fs=10, fw="bold")
txt(FU_X, 7.25, "(Softmax-Gated Weights)", fs=8, style="italic")

# Alpha labels
for label, yy in [("a_img", Y_TOP), ("a_tex", Y_MID), ("a_col", Y_BOT)]:
    txt(FU_X-0.72, yy, label, fs=9, style="italic", color=C_TEXT)

# Arrows from feat bars to fusion (to alpha labels)
for yy in [Y_TOP, Y_MID, Y_BOT]:
    arrow(FV_X+0.15, yy, FU_X-FU_W/2+0.05, yy)

# Softmax circle
circle = plt.Circle((FU_X+0.15, 5.0), 0.55, color="#D6EAF8",
                     ec=C_EDGE, lw=1.3, zorder=5)
ax.add_patch(circle)
txt(FU_X+0.15, 5.08, "Softmax", fs=7.5, fw="bold")
txt(FU_X+0.15, 4.72, r"$\Sigma\alpha_i=1$", fs=8)

# Arrows alpha → softmax
for yy in [Y_TOP, Y_MID, Y_BOT]:
    arrow(FU_X-0.35, yy, FU_X-0.38, 5.0+np.sign(yy-5.0)*0.52)

# Multiply symbols (⊗)
MUL_X = FU_X + 0.95
for yy in [Y_TOP, Y_MID, Y_BOT]:
    circle2 = plt.Circle((MUL_X, yy), 0.22, color="white",
                          ec=C_EDGE, lw=1.2, zorder=5)
    ax.add_patch(circle2)
    txt(MUL_X, yy, "x", fs=10, fw="bold")   # × symbol
    arrow(FU_X+0.15+0.55, 5.0+np.sign(yy-5.0)*0.03,
          MUL_X-0.22, yy, fc=C_EDGE, lw=1.2, ms=10)

# ── Fused feature bars ────────────────────────────────────────────────────────
FB_X = 11.35
# stacked colored bars (blue top, orange mid, green bot)
for color, cy, h in [(C_CNN, 6.4, 2.4), (C_TEX, 4.7, 1.5), (C_COL, 3.5, 0.9)]:
    feat_bars(FB_X, cy, h, color, n=5, w=0.28)

arrow(MUL_X+0.22, Y_TOP, FB_X-0.2, 6.6, ms=10)
arrow(MUL_X+0.22, Y_MID, FB_X-0.2, 4.8, ms=10)
arrow(MUL_X+0.22, Y_BOT, FB_X-0.2, 3.6, ms=10)

txt(FB_X, 2.5, "2176-d", fs=8.5, fw="bold", color=C_EDGE)

# ── FUSION HEAD ───────────────────────────────────────────────────────────────
FH_X = 13.5; FH_Y = 5.0; FH_W = 2.6; FH_H = 3.2
rbox(FH_X, FH_Y, FH_W, FH_H, C_HEAD)
txt(FH_X, FH_Y+1.35, "fusion_head", fs=10, fw="bold")
txt(FH_X, FH_Y+1.0,  "MLP", fs=9)
txt(FH_X, FH_Y+0.65, "(2176 → 512 → 11)", fs=8.5)

# Draw simple NN dots inside
for layer_x, n_dots in [(FH_X-0.65, 4), (FH_X, 3), (FH_X+0.65, 3)]:
    for j in range(n_dots):
        cy2 = FH_Y - 0.4 + j*0.28 - (n_dots-1)*0.14
        c3 = plt.Circle((layer_x, cy2), 0.07, color="white",
                         ec=C_EDGE, lw=1, zorder=6)
        ax.add_patch(c3)

# NN connection lines
for j in range(4):
    cy_l = FH_Y - 0.4 + j*0.28 - 3*0.14
    for k in range(3):
        cy_r = FH_Y - 0.4 + k*0.28 - 2*0.14
        ax.plot([FH_X-0.58, FH_X-0.07], [cy_l, cy_r],
                color=C_EDGE, lw=0.5, alpha=0.5, zorder=5)
        ax.plot([FH_X+0.07, FH_X+0.58], [cy_r, cy_r],
                color=C_EDGE, lw=0.5, alpha=0.5, zorder=5)

# dots label
txt(FH_X, FH_Y-0.65, "...", fs=12, color=C_EDGE)

arrow(FB_X+0.22, 5.0, FH_X-FH_W/2, 5.0)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUT_X = 16.5; OUT_Y = 5.0
rbox(OUT_X, OUT_Y, 2.0, 2.0, C_OUT)
txt(OUT_X, OUT_Y+0.52, "11-class", fs=10, fw="bold")
txt(OUT_X, OUT_Y+0.15, "output", fs=10, fw="bold")

# mini bar chart inside output box
bar_vals = [0.99, 0.3, 0.15, 0.08, 0.05, 0.04]
bar_colors = [C_CNN, C_TEX, C_COL, C_FUSE, C_HEAD, C_EDGE]
for i, (bv, bc) in enumerate(zip(bar_vals, bar_colors)):
    bx = OUT_X - 0.65 + i*0.25
    bar_h2 = bv * 0.5
    p = FancyBboxPatch((bx-0.09, OUT_Y-0.65), 0.18, bar_h2,
                       boxstyle="round,pad=0,rounding_size=0.04",
                       facecolor=bc, edgecolor="white", lw=0.5, zorder=5)
    ax.add_patch(p)

arrow(FH_X+FH_W/2, FH_Y, OUT_X-1.0, OUT_Y)

# ── FEATURE DIMENSIONS BOX (dashed, top right) ────────────────────────────────
dx, dy, dw, dh = 14.4, 8.8, 3.3, 2.6
p = FancyBboxPatch((dx-dw/2, dy-dh/2), dw, dh,
                   boxstyle="round,pad=0,rounding_size=0.3",
                   facecolor="white", edgecolor=C_DASH,
                   linewidth=1.5, linestyle="--", zorder=4)
ax.add_patch(p)
txt(dx, dy+1.02, "Feature Dimensions", fs=9.5, fw="bold", color=C_TEXT)
for i, line in enumerate([
    "EfficientNet-B4 (Image):  1792-d",
    "Texture MLP:                    256-d",
    "Color MLP:                        128-d",
    "Concatenated:                  2176-d",
    "Output Classes:                      11",
]):
    txt(dx, dy+0.55 - i*0.38, line, fs=8.2, ha="center", color=C_TEXT)

# ── LEGEND (bottom right, dashed) ────────────────────────────────────────────
lx, ly, lw2, lh2 = 14.5, 2.9, 3.3, 2.2
p2 = FancyBboxPatch((lx-lw2/2, ly-lh2/2), lw2, lh2,
                    boxstyle="round,pad=0,rounding_size=0.3",
                    facecolor="white", edgecolor=C_DASH,
                    linewidth=1.5, linestyle="--", zorder=4)
ax.add_patch(p2)
txt(lx, ly+0.78, "Legend", fs=9.5, fw="bold")
legend_items = [
    (C_CNN,  "Image Branch (EfficientNet-B4)"),
    (C_TEX,  "Texture Branch (MLP)"),
    (C_COL,  "Color Branch (MLP)"),
]
for i, (fc, lb) in enumerate(legend_items):
    yy = ly+0.35 - i*0.42
    rbox(lx-1.1, yy, 0.38, 0.22, fc, ec=C_EDGE, lw=0.8, rad=0.06)
    txt(lx-0.04, yy, lb, fs=8, ha="center", color=C_TEXT)

txt(lx-1.1, ly+0.35-3*0.42, "x", fs=11, fw="bold")
txt(lx-0.04, ly+0.35-3*0.42, "Element-wise multiplication", fs=8, color=C_TEXT)
txt(lx-1.1,  ly+0.35-4*0.42, r"$\Sigma$", fs=11)
txt(lx-0.04, ly+0.35-4*0.42, "Summation", fs=8, color=C_TEXT)

plt.tight_layout(pad=0.1)
out_dir = Path("D:/SpiceNet/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "arch_figure.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
plt.close()
print(f"Saved -> {out_path}")
