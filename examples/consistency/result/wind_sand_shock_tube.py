#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT/"wind_sand_shock_tube.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

def values(name):
    return [float(row[name]) for row in rows]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

x = values("x_m")

panels = [
    ("rho_g", r"Apparent gas density $\rho_g$ (-)"),
    ("rho_p", r"Apparent particle density $\rho_p$ (-)"),
    ("u_g",   r"Gas velocity $U_g$ (-)"),
    ("u_p",   r"Particle velocity $U_p$ (-)"),
    ("p_g",   r"Gas pressure $p_g$ (-)"),
    ("T_p",   r"Particle temperature $T_p$ (-)"),
]

fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharex=True)

for ax, (name, ylabel) in zip(axes.flat, panels):
    ax.plot(
        x, values(name + "_published"),
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Literature UGKWP",
    )
    ax.plot(
        x, values(name + "_GPU_GKS_GKP"),
        color="#C44E52",
        linestyle="-",
        linewidth=1.1,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#C44E52",
        markeredgewidth=0.7,
        markersize=3.0,
        markevery=4,
        label="GPU-GKS-GKP",
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(top=True, right=True, length=5, width=0.9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

for ax in axes[-1]:
    ax.set_xlabel(r"Axial position $x$ (-)")

axes[0, 0].legend(frameon=False, loc="best")

fig.tight_layout(pad=0.7, h_pad=0.35, w_pad=1.0)
fig.savefig(ROOT/"wind_sand_shock_tube.png", dpi=600, facecolor="white")
plt.close(fig)
