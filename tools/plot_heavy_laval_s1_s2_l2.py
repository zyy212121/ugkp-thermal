#!/usr/bin/env python3

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


root = Path(__file__).resolve().parent
with (root / "heavy_laval_s1_s2_l2.csv").open(encoding="utf-8", newline="") as stream:
    data = list(csv.DictReader(stream))

font_path = Path(os.environ.get("UGKP_CHINESE_FONT", ""))
chinese = FontProperties(fname=str(font_path), size=15) if font_path.is_file() else None
label_font = FontProperties(fname=str(font_path), size=18) if font_path.is_file() else None
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

x = [int(row["target_cell_initial_parcels"]) for row in data]
series = (
    ("S1", "s1", "black", "--", "s"),
    ("原 S2", "legacy_s2", "#4C72B0", "-.", "^"),
    ("统一 L2", "unified_l2", "#C44E52", "-", "o"),
)
panels = (
    ("total", "单步总时间 (ms)"),
    ("collision_pool", "碰撞池规约时间 (ms)"),
    ("moments", "宏观矩恢复时间 (ms)"),
)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (field, ylabel) in zip(axes, panels):
    for name, prefix, color, linestyle, marker in series:
        y = [float(row[f"{prefix}_{field}_ms"]) for row in data]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.6, marker=marker, markersize=5.0, markerfacecolor="white", markeredgewidth=0.9, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("目标单元初始颗粒数", fontproperties=label_font)
    ax.set_ylabel(ylabel, fontproperties=label_font)
    ax.set_xticks(x, ["11k", "22k", "44k", "88k", "176k", "352k", "704k"])
    ax.tick_params(top=True, right=True, length=6, width=1.0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
axes[0].legend(frameon=False, loc="upper left", prop=chinese)
fig.tight_layout(pad=0.7, w_pad=1.0)
fig.savefig(root / "heavy_laval_s1_s2_l2.png", dpi=600, facecolor="white")
plt.close(fig)
