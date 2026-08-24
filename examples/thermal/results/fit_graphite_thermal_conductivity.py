#!/usr/bin/env python3
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "graphite_diffusivity_blue_points.csv"
TABLE = ROOT / "graphite_thermal_conductivity_fit.csv"
FIGURE = ROOT / "graphite_thermal_conductivity_fit.png"
RHO = 1800.0


def cp(temperature):
    return np.full_like(np.asarray(temperature, dtype=float), 710.0)


raw = np.genfromtxt(RAW, delimiter=",", names=True)
temperature = raw["temperature_K"]
diffusivity = raw["thermal_diffusivity_m2_s"]
scaled_diffusivity = diffusivity/1.0e-5
fit = UnivariateSpline(temperature, scaled_diffusivity, s=0.4)
table_temperature = np.arange(315.0, 816.0, 25.0)
table_diffusivity = fit(table_temperature)*1.0e-5
table_cp = cp(table_temperature)
table_kappa = RHO*table_cp*table_diffusivity
raw_kappa = RHO*cp(temperature)*diffusivity

with TABLE.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("temperature_K", "smoothed_thermal_diffusivity_m2_s", "Cp_J_kg_K", "thermal_conductivity_W_m_K"))
    for values in zip(table_temperature, table_diffusivity, table_cp, table_kappa):
        writer.writerow((f"{values[0]:.0f}", f"{values[1]:.12e}", f"{values[2]:.8f}", f"{values[3]:.8f}"))

fine_temperature = np.linspace(table_temperature[0], table_temperature[-1], 600)
fine_diffusivity = fit(fine_temperature)*1.0e-5
fine_kappa = RHO*cp(fine_temperature)*fine_diffusivity
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
fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
axis.scatter(temperature, raw_kappa, color="#4C72B0", s=34, label="Digitized blue points", zorder=3)
axis.plot(fine_temperature, fine_kappa, color="#C44E52", linewidth=2.6, label="Smoothed conductivity")
axis.plot(table_temperature, table_kappa, linestyle="none", marker="o", markersize=5,
          markerfacecolor="white", markeredgecolor="#C44E52", markeredgewidth=0.9,
          label="Solver table")
axis.set_xlabel("Temperature $T$ (K)")
axis.set_ylabel("Thermal conductivity $k$ (W m$^{-1}$ K$^{-1}$)")
axis.grid(False)
axis.tick_params(top=True, right=True, length=6, width=1.0)
axis.legend(frameon=False)
for spine in axis.spines.values():
    spine.set_linewidth(1.0)
fig.tight_layout()
fig.savefig(FIGURE, dpi=600, facecolor="white")
plt.close(fig)
print(f"wrote={TABLE}")
print(f"wrote={FIGURE}")
