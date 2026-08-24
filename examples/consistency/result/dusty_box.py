#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT/"dusty_box.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
def values(name):
    return [float(row[name]) for row in rows]
plt.rcParams.update({"font.family":"serif","font.size":18,"axes.labelsize":22,"legend.fontsize":15,"xtick.labelsize":18,"ytick.labelsize":18,"axes.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","mathtext.fontset":"stix","axes.unicode_minus":False})
t = values("time_s")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
for ax, phase, exact, numerical, color, marker in ((axes[0],"Gas phase","Ug_analytic_m_s","Ug_GPU_GKS_GKP_m_s","#C44E52","o"),(axes[1],"Particle phase","Up_analytic_m_s","Up_GPU_GKS_GKP_m_s","#4C72B0","s")):
    ax.plot(t, values(exact), color="black", linewidth=2.6, label="Analytical solution")
    ax.plot(t, values(numerical), color=color, linewidth=1.6, marker=marker, markerfacecolor="white", markeredgewidth=0.9, markersize=4.8, markevery=2, label="GPU-GKS-GKP")
    ax.set_xlabel(r"Time $t$ ($\mathrm{s}$)")
    ax.set_ylabel(rf"{phase} axial velocity $U_x$ ($\mathrm{{m\,s^{{-1}}}}$)")
    ax.tick_params(top=True, right=True, length=6, width=1.0)
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(1.0)
axes[0].legend(frameon=False, loc="best")
fig.tight_layout(pad=0.7, w_pad=1.0)
fig.savefig(ROOT/"dusty_box.png", dpi=600, facecolor="white")
plt.close(fig)
