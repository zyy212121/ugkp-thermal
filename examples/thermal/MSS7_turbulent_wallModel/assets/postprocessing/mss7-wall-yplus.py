import csv
from pathlib import Path

import matplotlib.pyplot as plt


CASE_ROOT = Path(__file__).resolve().parents[2]
ROOT = CASE_ROOT.parent / "results" / CASE_ROOT.name
BASE = ROOT/"data"/Path(__file__).stem
FIGURE = ROOT/"figures"/f"{Path(__file__).stem}.png"

with BASE.with_suffix(".csv").open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))

x = [float(row["x_m"]) for row in rows]
y_plus = [float(row["y_plus"]) for row in rows]

if not x or len(x) != len(y_plus):
    raise ValueError("The input CSV does not contain a valid x_m/y_plus series")
if any(x[index] <= x[index - 1] for index in range(1, len(x))):
    raise ValueError("x_m must be strictly increasing")

plt.rcParams.update(
    {
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
    }
)

fig, ax = plt.subplots(figsize=(10.0, 8.0), facecolor="white")
ax.plot(
    x,
    y_plus,
    color="#C44E52",
    linestyle="-",
    linewidth=1.6,
    marker="o",
    markerfacecolor="white",
    markeredgewidth=0.9,
    markersize=5.0,
    markevery=8,
)
ax.set_xlabel(r"Axial coordinate, $x$ (m)")
ax.set_ylabel(r"Wall coordinate, $y^+$")
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=20.0)
ax.set_facecolor("white")
ax.grid(False)
ax.tick_params(top=True, right=True, length=6, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

fig.tight_layout(pad=0.7)
fig.savefig(FIGURE, dpi=600, facecolor="white")
plt.close(fig)
