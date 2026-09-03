#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "result"
with (ROOT/"sod_shock_tube.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
def values(name):
    return [float(row[name]) for row in rows]
plt.rcParams.update({"font.family":"serif","font.size":18,"axes.labelsize":22,"legend.fontsize":15,"xtick.labelsize":18,"ytick.labelsize":18,"axes.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","mathtext.fontset":"stix","axes.unicode_minus":False})
x = values("x_m")
series = (("rho_exact_kg_m3","rho_GPU_GKS_GKP_kg_m3",r"Density $\rho$ ($\mathrm{kg\,m^{-3}}$)"),("Ux_exact_m_s","Ux_GPU_GKS_GKP_m_s",r"Axial velocity $U_x$ ($\mathrm{m\,s^{-1}}$)"),("p_exact_Pa","p_GPU_GKS_GKP_Pa",r"Pressure $p$ ($\mathrm{Pa}$)"))
fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
for ax, (exact, numerical, ylabel) in zip(axes, series):
    ax.plot(x, values(exact), color="black", linewidth=2.6, label="Exact solution")
    ax.plot(x, values(numerical), color="#C44E52", linewidth=1.6, marker="o", markerfacecolor="white", markeredgewidth=0.9, markersize=4.8, markevery=8, label="GPU-GKS-GKP")
    ax.set_xlabel(r"Axial position $x$ ($\mathrm{m}$)")
    ax.set_ylabel(ylabel)
    ax.tick_params(top=True, right=True, length=6, width=1.0)
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(1.0)
axes[0].legend(frameon=False, loc="best")
fig.tight_layout(pad=0.7, w_pad=1.0)
fig.savefig(ROOT/"sod_shock_tube.png", dpi=600, facecolor="white")
plt.close(fig)
