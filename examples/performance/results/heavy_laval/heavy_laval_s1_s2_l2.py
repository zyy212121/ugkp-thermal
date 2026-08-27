#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
with (root / "heavy_laval_s1_s2_l2.csv").open(encoding="utf-8", newline="") as stream:
    data = list(csv.DictReader(stream))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 22,
    "legend.fontsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

x = [int(row["target_cell_initial_parcels"]) / 1000.0 for row in data]
series = (
    ("S1", "s1", "#C44E52", "-", "o"),
    ("S2", "legacy_s2", "#4C72B0", "--", "s"),
    ("L2", "unified_l2", "#55A868", "-.", "^"),
)


def finish_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, ["11", "22", "44", "88", "176", "352", "704"])
    ax.set_xlabel(r"Initial refined-cell occupancy $n_{c,0}$ ($10^3$ parcels)")
    ax.tick_params(top=True, right=True, length=6, width=1.0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def errorbar(ax, prefix, field, label, color, linestyle, marker):
    y = [float(row[f"{prefix}_{field}_ms"]) for row in data]
    yerr = [float(row[f"{prefix}_{field}_std_ms"]) for row in data]
    ax.errorbar(
        x, y, yerr=yerr, color=color, linestyle=linestyle, linewidth=1.6,
        marker=marker, markersize=5.0, markerfacecolor="white",
        markeredgewidth=0.9, capsize=3.0, label=label,
    )


fig, ax = plt.subplots(figsize=(10.0, 8.0))
for label, prefix, color, linestyle, marker in series:
    errorbar(ax, prefix, "total", label, color, linestyle, marker)
finish_axis(ax)
ax.set_ylabel("Per-step total time (ms)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.7)
fig.savefig(root / "heavy_laval_total_time.png", dpi=600, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
for ax, field, title in zip(
    axes,
    ("collision_pool", "moments"),
    ("(a) Collision pool", "(b) Moment reduction"),
):
    for label, prefix, color, linestyle, marker in series:
        errorbar(ax, prefix, field, label, color, linestyle, marker)
    finish_axis(ax)
    ax.set_ylabel("Per-step kernel time (ms)")
    ax.text(0.5, 0.94, title, transform=ax.transAxes, ha="center", va="top")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.7, w_pad=1.0)
fig.savefig(root / "heavy_laval_stage_cost.png", dpi=600, facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.0, 8.0))
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Break-even")
for label, field, color, linestyle, marker in (
    (r"$T_{S1}/T_{L2}$", "s1", "#C44E52", "-", "o"),
    (r"$T_{S2}/T_{L2}$", "legacy_s2", "#4C72B0", "--", "s"),
):
    y = [float(row[f"l2_speedup_over_{field}"]) for row in data]
    yerr = [float(row[f"l2_speedup_over_{field}_std"]) for row in data]
    ax.errorbar(
        x, y, yerr=yerr, color=color, linestyle=linestyle, linewidth=1.6,
        marker=marker, markersize=5.0, markerfacecolor="white",
        markeredgewidth=0.9, capsize=3.0, label=label,
    )
finish_axis(ax)
ax.set_ylabel("Speedup")
ax.legend(frameon=False, loc="best")
fig.tight_layout(pad=0.7)
fig.savefig(root / "heavy_laval_speedup.png", dpi=600, facecolor="white")
plt.close(fig)
