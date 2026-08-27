#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/heavy_laval_scheduler_scan_20260820"
OUTPUT_ROOT = RESULT_ROOT / "figure"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
BASE = OUTPUT_ROOT / "heavy_laval_scheduler_scan"

VARIANTS = (
    ("Dynamic q=1", "q1_shared", "#C44E52", "-", "o"),
    ("Dynamic q=2", "q2", "#4C72B0", "--", "s"),
    ("Dynamic q=4", "q4", "#55A868", "-.", "^"),
    ("Static moments", "static_moments", "#DD8452", ":", "D"),
    ("Source unswitch", "source_unswitch", "#6B7A8F", (0, (5, 2)), "v"),
)
FACTORS = (1, 2)


def measured_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if int(row["measured"]) == 1]


records = []
for label, directory, _, _, _ in VARIANTS:
    rows = measured_rows(RESULT_ROOT / directory / "process_medians.csv")
    for factor in FACTORS:
        selected = [row for row in rows if int(row["factor"]) == factor]
        total = [float(row["total_ms"]) for row in selected]
        pool = [float(row["collision_pool_ms"]) for row in selected]
        moments = [float(row["moments_ms"]) for row in selected]
        particles = [float(row["particle_count"]) for row in selected]
        records.append({
            "variant": label,
            "factor": factor,
            "target_cell_initial_parcels": 11000 * factor,
            "process_particle_count": statistics.median(particles),
            "total_ms": statistics.median(total),
            "total_std_ms": statistics.stdev(total),
            "collision_pool_ms": statistics.median(pool),
            "moments_ms": statistics.median(moments),
        })

q1 = {
    int(row["factor"]): float(row["total_ms"])
    for row in records
    if row["variant"] == "Dynamic q=1"
}
for row in records:
    row["q1_speedup"] = q1[int(row["factor"])] / float(row["total_ms"])

with BASE.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

with BASE.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
    data = list(csv.DictReader(stream))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 22,
    "legend.fontsize": 13,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def finish_axis(axis):
    axis.set_xscale("log", base=2)
    counts = sorted({float(row["target_cell_initial_parcels"]) for row in data})
    axis.set_xticks(counts, [f"{value / 1.0e3:.0f}" for value in counts])
    axis.set_xlabel(r"Particle count $N_p$ ($10^3$ parcels)")
    axis.tick_params(top=True, right=True, length=6, width=1.0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
for label, _, color, linestyle, marker in VARIANTS:
    subset = sorted(
        (row for row in data if row["variant"] == label),
        key=lambda row: float(row["target_cell_initial_parcels"]),
    )
    x = [float(row["target_cell_initial_parcels"]) for row in subset]
    axes[0].errorbar(
        x,
        [float(row["total_ms"]) for row in subset],
        yerr=[float(row["total_std_ms"]) for row in subset],
        color=color,
        linestyle=linestyle,
        linewidth=1.6,
        marker=marker,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        capsize=3.0,
        label=label,
    )
    axes[1].plot(
        x,
        [float(row["q1_speedup"]) for row in subset],
        color=color,
        linestyle=linestyle,
        linewidth=1.6,
        marker=marker,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=label,
    )

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.5)
finish_axis(axes[0])
finish_axis(axes[1])
axes[0].set_ylabel("Per-step total time (ms)")
axes[1].set_ylabel(r"Relative performance $T_{q=1}/T$")
axes[0].text(0.5, 0.95, "(a) Total time", transform=axes[0].transAxes, ha="center", va="top")
axes[1].text(0.5, 0.95, "(b) Relative performance", transform=axes[1].transAxes, ha="center", va="top")
axes[0].legend(frameon=False, loc="best")
fig.tight_layout(pad=0.7, w_pad=1.0)
fig.savefig(BASE.with_suffix(".png"), dpi=600, facecolor="white")
plt.close(fig)

print(BASE.with_suffix(".csv"))
print(BASE.with_suffix(".png"))
