#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "result"
with (ROOT/"dusty_box.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
def values(name):
    return [float(row[name]) for row in rows]
plt.rcParams.update({"font.family":"serif","font.size":18,"axes.labelsize":22,"legend.fontsize":15,"xtick.labelsize":18,"ytick.labelsize":18,"axes.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","mathtext.fontset":"stix","axes.unicode_minus":False})
t = values("time_s")
fig, ax = plt.subplots(figsize=(7.4, 5.6))
ax.plot(t, values("Ug_analytic_m_s"), color="black", linewidth=2.6, label="Gas analytical")
ax.plot(t, values("Ug_GPU_GKS_GKP_m_s"), color="#C44E52", linewidth=1.6, marker="o", markerfacecolor="white", markeredgewidth=0.9, markersize=4.8, markevery=2, label="Gas numerical")
ax.plot(t, values("Up_analytic_m_s"), color="black", linewidth=2.6, linestyle="--", label="Particle analytical")
ax.plot(t, values("Up_GPU_GKS_GKP_m_s"), color="#4C72B0", linewidth=1.6, marker="s", markerfacecolor="white", markeredgewidth=0.9, markersize=4.8, markevery=2, label="Particle numerical")
ax.set_xlabel(r"Time $t$ ($\mathrm{s}$)")
ax.set_ylabel(r"Axial velocity $U_x$ ($\mathrm{m\,s^{-1}}$)", fontsize=20)
ax.ticklabel_format(axis="x", style="sci", scilimits=(-4, -4), useMathText=True)
ax.tick_params(top=True, right=True, length=6, width=1.0)
ax.grid(False)
for spine in ax.spines.values(): spine.set_linewidth(1.0)
ax.legend(frameon=False, loc="center right")
fig.tight_layout(pad=0.7)
fig.savefig(ROOT/"dusty_box.png", dpi=600, facecolor="white")
plt.close(fig)
