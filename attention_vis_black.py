# attention_vis_black.py
"""
Generates two 6×6 heat-maps that compare baseline attention with logic-masked
attention.  Figures use a fully black background, white text, and white–outlined
squares to highlight the forbidden Drug→Gene cells.

Outputs
-------
baseline_black.png
logic_black.png
"""

from __future__ import annotations
import math, random, numpy as np
import torch, matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1.  Styling defaults (dark background, nicer fonts)                         #
# --------------------------------------------------------------------------- #
rcParams.update({
    "figure.facecolor"  : "black",
    "axes.facecolor"    : "black",
    "savefig.facecolor" : "black",
    "text.color"        : "white",
    "axes.labelcolor"   : "white",
    "axes.edgecolor"    : "white",
    "xtick.color"       : "white",
    "ytick.color"       : "white",
    "font.size"         : 12,
    "axes.titleweight"  : "bold",
    "axes.titlesize"    : 14,
    "image.cmap"        : "viridis",   # good contrast on dark bg
})

# --------------------------------------------------------------------------- #
# 2.  Tiny heterogeneous graph + random attention                             #
# --------------------------------------------------------------------------- #
SEED = 1
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

n, d = 6, 8
x     = torch.randn(n, d)
types = torch.tensor([0, 1, 0, 2, 1, 0])   # 0 Protein, 1 Drug, 2 Gene
DRUG, GENE = 1, 2

# Hard mask (Drug → Gene forbidden)
mask = torch.zeros(n, n)
mask[(types == DRUG).nonzero(), (types == GENE).nonzero()] = -1e9

# Single-head random attention
Q = torch.randn_like(x); K = torch.randn_like(x)
scores_base  = (Q @ K.T) / math.sqrt(d)
att_base     = scores_base.softmax(-1)

scores_logic = scores_base + mask
att_logic    = scores_logic.softmax(-1)

# Common colour scale across the two plots
vmin, vmax = 0.0, att_base.max().item()

# Helper to draw one heat-map -------------------------------------------------
def draw(att: torch.Tensor, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    im = ax.imshow(att, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Key (source node)")
    ax.set_ylabel("Query (target node)")
    # White outline on forbidden cells
    rows = (types == DRUG).nonzero(as_tuple=False).view(-1)
    col  = (types == GENE).nonzero(as_tuple=False).item()
    ax.scatter([col]*len(rows), rows, marker="s",
               facecolors="none", edgecolors="white",
               linewidths=1.8, s=120)
    # Colour-bar also on black bg
    cbar = fig.colorbar(im, fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# 3.  Save the two figures                                                    #
# --------------------------------------------------------------------------- #
out_dir = Path(".")
draw(att_base.detach(),  "Baseline attention",   out_dir / "baseline_black.png")
draw(att_logic.detach(), "With logic adapter",   out_dir / "logic_black.png")

print("Saved: baseline_black.png  logic_black.png")
