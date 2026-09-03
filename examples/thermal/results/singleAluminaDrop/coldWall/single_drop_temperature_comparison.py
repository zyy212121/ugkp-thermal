import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BASE = ROOT/"data"/Path(__file__).stem
FIGURE = ROOT/"figures"/f"{Path(__file__).stem}.png"

with BASE.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))


def series(record_type, field):
    selected = [
        (float(row["time_ms"]), float(row[field]))
        for row in rows
        if row["record_type"] == record_type and row.get(field, "") != ""
    ]
    return [item[0] for item in selected], [item[1] for item in selected]


experiment_time, experiment_temperature = series("experiment", "experiment_surface_temperature_C")
cold_time, cold_temperature = series("coldWall", "cold_free_surface_temperature_C")
cold_contact_time, cold_contact_temperature = series("coldWall", "cold_wall_contact_layer_temperature_C")
cold_volume_time, cold_volume_temperature = series("coldWall", "cold_axial_node_mean_temperature_C")

plt.rcParams.update({
    "font.family": "serif", "font.size": 18, "axes.labelsize": 22,
    "legend.fontsize": 15, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
})
fig, ax = plt.subplots(figsize=(10.0, 8.0), facecolor="white")
ax.plot(experiment_time, experiment_temperature, color="black", linestyle="-",
        linewidth=2.6, label="Experiment")
ax.plot(cold_time, cold_temperature, color="#C44E52", linestyle="-",
        linewidth=1.6, marker="o", markerfacecolor="white", markeredgewidth=0.9,
        markersize=5.0, markevery=8, label="Calculated (free surface)")
ax.plot(cold_contact_time, cold_contact_temperature, color="#DD8452",
        linestyle="-.", linewidth=1.6, marker="^", markerfacecolor="white",
        markeredgewidth=0.9, markersize=5.0, markevery=8,
        label="Calculated (contact layer)")
ax.plot(cold_volume_time, cold_volume_temperature, color="#55A868",
        linestyle=":", linewidth=1.6, marker="D", markerfacecolor="white",
        markeredgewidth=0.9, markersize=4.8, markevery=8,
        label="Calculated (volume mean)")
ax.set_xlabel(r"Time after impact $t$ (ms)")
ax.set_ylabel(r"Temperature $T$ ($^{\circ}$C)")
ax.set_xlim(0.0, 10.0)
ax.set_facecolor("white")
ax.grid(False)
ax.tick_params(top=True, right=True, length=6, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
ax.legend(frameon=False, loc="best", fontsize=14, handlelength=2.4, labelspacing=0.45)
fig.tight_layout(pad=0.7)
fig.savefig(FIGURE, dpi=600, facecolor="white")
plt.close(fig)
