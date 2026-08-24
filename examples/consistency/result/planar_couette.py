#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT/"planar_couette.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
def values(name):
    return [float(row[name]) for row in rows]
plt.rcParams.update({"font.family":"serif","font.size":18,"axes.labelsize":22,"legend.fontsize":15,"xtick.labelsize":18,"ytick.labelsize":18,"axes.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","mathtext.fontset":"stix","axes.unicode_minus":False})
fig, ax = plt.subplots(figsize=(10.0, 8.0))
ax.plot(values("Ux_analytic_m_s"), values("y_m"), color="black", linewidth=2.6, label="Analytical solution")
ax.plot(values("Ux_GPU_GKS_GKP_m_s"), values("y_m"), color="#C44E52", linewidth=1.6, marker="o", markerfacecolor="white", markeredgewidth=0.9, markersize=5.0, markevery=4, label="GPU-GKS-GKP")
ax.set_xlabel(r"Axial velocity $U_x$ ($\mathrm{m\,s^{-1}}$)")
ax.set_ylabel(r"Wall-normal position $y$ ($\mathrm{m}$)")
ax.tick_params(top=True, right=True, length=6, width=1.0)
ax.grid(False)
for spine in ax.spines.values(): spine.set_linewidth(1.0)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig(ROOT/"planar_couette.png", dpi=600, facecolor="white")
plt.close(fig)
