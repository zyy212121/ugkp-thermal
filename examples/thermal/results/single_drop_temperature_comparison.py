import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).with_suffix("")


def values(rows, x_name, y_name):
    pairs = [
        (float(row[x_name]), float(row[y_name]))
        for row in rows
        if row.get(x_name, "") != "" and row.get(y_name, "") != ""
    ]
    return [item[0] for item in pairs], [item[1] for item in pairs]


with BASE.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

experiment_time, experiment_temperature = values(
    rows, "experiment_time_ms", "experiment_surface_temperature_C"
)
hot_time, hot_temperature = values(
    rows, "hot_time_ms", "hot_lumped_temperature_C"
)
cold_time, cold_temperature = values(
    rows, "cold_time_ms", "cold_free_surface_temperature_C"
)
cold_contact_time, cold_contact_temperature = values(
    rows, "cold_time_ms", "cold_wall_contact_layer_temperature_C"
)
cold_volume_time, cold_volume_temperature = values(
    rows, "cold_time_ms", "cold_axial_node_mean_temperature_C"
)

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
    experiment_time,
    experiment_temperature,
    color="black",
    linestyle="-",
    linewidth=2.6,
    label="Experiment",
)
ax.plot(
    hot_time,
    hot_temperature,
    color="#4C72B0",
    linestyle="--",
    linewidth=1.6,
    marker="s",
    markerfacecolor="white",
    markeredgewidth=0.9,
    markersize=5.0,
    markevery=8,
    label="hotWall",
)
ax.plot(
    cold_time,
    cold_temperature,
    color="#C44E52",
    linestyle="-",
    linewidth=1.6,
    marker="o",
    markerfacecolor="white",
    markeredgewidth=0.9,
    markersize=5.0,
    markevery=8,
    label="coldWall1D free surface",
)
ax.plot(
    cold_contact_time,
    cold_contact_temperature,
    color="#DD8452",
    linestyle="-.",
    linewidth=1.6,
    marker="^",
    markerfacecolor="white",
    markeredgewidth=0.9,
    markersize=5.0,
    markevery=8,
    label="coldWall1D wall contact",
)
ax.plot(
    cold_volume_time,
    cold_volume_temperature,
    color="#55A868",
    linestyle=":",
    linewidth=1.6,
    marker="D",
    markerfacecolor="white",
    markeredgewidth=0.9,
    markersize=4.8,
    markevery=8,
    label="coldWall1D volume mean",
)
ax.set_xlabel(r"Time after impact $t$ (ms)")
ax.set_ylabel(r"Temperature $T$ ($^{\circ}$C)")
ax.set_xlim(0.0, 10.0)
ax.set_facecolor("white")
ax.grid(False)
ax.tick_params(top=True, right=True, length=6, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
ax.legend(
    frameon=False,
    loc="best",
    ncol=1,
    fontsize=14,
    columnspacing=0.9,
    handlelength=2.4,
    labelspacing=0.45,
)
fig.tight_layout(pad=0.7)
fig.savefig(BASE.with_suffix(".png"), dpi=600, facecolor="white")
plt.close(fig)
