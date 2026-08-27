import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).with_suffix("")
with BASE.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))


def series(record_type, field):
    selected = [
        (float(row["time_ms"]), float(row[field]))
        for row in rows
        if row["record_type"] == record_type and row.get(field, "") != ""
    ]
    return [item[0] for item in selected], [item[1] for item in selected]


experiment_time, experiment_beta = series("experiment", "experiment_beta")
cold_time, cold_beta = series("coldWall", "cold_beta")

plt.rcParams.update({
    "font.family": "serif", "font.size": 18, "axes.labelsize": 22,
    "legend.fontsize": 15, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
})
fig, ax = plt.subplots(figsize=(10.0, 8.0), facecolor="white")
ax.plot(experiment_time, experiment_beta, color="black", linestyle="-",
        linewidth=2.6, label="Experiment")
ax.plot(cold_time, cold_beta, color="#C44E52", linestyle="-", linewidth=1.6,
        marker="o", markerfacecolor="white", markeredgewidth=0.9,
        markersize=5.0, markevery=8, label="Calculated")
ax.set_xlabel(r"Time after impact $t$ (ms)")
ax.set_ylabel(r"Spreading factor $\beta$")
ax.set_xlim(0.0, 10.0)
ax.set_ylim(bottom=0.0)
ax.set_facecolor("white")
ax.grid(False)
ax.tick_params(top=True, right=True, length=6, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
ax.legend(frameon=False, loc="best")
fig.tight_layout(pad=0.7)
fig.savefig(BASE.with_suffix(".png"), dpi=600, facecolor="white")
plt.close(fig)
