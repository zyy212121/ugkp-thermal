#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent.parent / "result"
CASE = ROOT.parent / "dustyWave"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def scalar_field(path: Path, size: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s+({NUMBER})\s*;", text)
    if uniform:
        return np.full(size, float(uniform.group(1)))
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Cannot read {path}")
    values = np.fromstring(match.group(2), sep=" ")
    if int(match.group(1)) != size or values.size != size:
        raise RuntimeError(f"Unexpected cell count in {path}")
    return values


def vector_field(path: Path, size: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)\s*;", text)
    if uniform:
        value = np.asarray([float(uniform.group(i)) for i in range(1, 4)])
        return np.repeat(value[None, :], size, axis=0)
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Cannot read {path}")
    rows = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", match.group(2))
    values = np.asarray(rows, dtype=float)
    if int(match.group(1)) != size or values.shape != (size, 3):
        raise RuntimeError(f"Unexpected cell count in {path}")
    return values


def reference(x: np.ndarray, time: float, p: dict[str, float]) -> dict[str, np.ndarray]:
    k = p["wave_number"]
    rho_g = p["rho_g0"]
    rho_p = p["rho_p0"]
    sound2 = p["gamma"] * p["p0"] / rho_g
    tau = p["response_time"]
    matrix = np.asarray(
        [
            [0, 0, -1j * k * rho_g, 0],
            [0, 0, 0, -1j * k * rho_p],
            [-1j * k * sound2 / rho_g, 0, -(rho_p / rho_g) / tau, (rho_p / rho_g) / tau],
            [0, 0, 1 / tau, -1 / tau],
        ],
        dtype=complex,
    )
    initial = -1j * p["amplitude"] * np.asarray([rho_g, rho_p, 1, 1], dtype=complex)
    perturbation = np.real((expm(matrix * time) @ initial)[:, None] * np.exp(1j * k * x)[None, :])
    return {
        "rho": rho_g + perturbation[0],
        "rhoP": rho_p + perturbation[1],
        "U": perturbation[2],
        "Us": perturbation[3],
        "p": p["p0"] + sound2 * perturbation[0],
    }


def relative_l2(numerical: np.ndarray, exact: np.ndarray, base: float) -> float:
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact - base))


def harmonic_error(x: np.ndarray, numerical: np.ndarray, exact: np.ndarray, base: float, k: float) -> tuple[float, float]:
    kernel = np.exp(-1j * k * x)
    numerical_hat = 2 * np.mean((numerical - base) * kernel)
    exact_hat = 2 * np.mean((exact - base) * kernel)
    return float(abs(abs(numerical_hat) - abs(exact_hat)) / abs(exact_hat)), float(abs(np.angle(numerical_hat / exact_hat)))


def main() -> None:
    p = json.loads((CASE / "reference/case_parameters.json").read_text(encoding="utf-8"))
    size = int(p["n_cells"])
    time = float(p["final_time"])
    final = CASE / f"{time:g}"
    x_all = (np.arange(size) + 0.5) * p["length"] / size
    begin, end = p["analysis_window"]
    mask = (x_all >= begin) & (x_all < end)
    x = x_all[mask]
    numerical = {
        "rho": scalar_field(final / "rho", size)[mask],
        "rhoP": scalar_field(final / "epsilonS", size)[mask] * p["rho_solid"],
        "U": vector_field(final / "U", size)[mask, 0],
        "Us": vector_field(final / "Us", size)[mask, 0],
        "p": scalar_field(final / "p", size)[mask],
    }
    exact_all = reference(x_all, time, p)
    exact = {name: values[mask] for name, values in exact_all.items()}
    bases = {"rho": p["rho_g0"], "rhoP": p["rho_p0"], "U": 0, "Us": 0, "p": p["p0"]}
    l2 = {name: relative_l2(numerical[name], exact[name], bases[name]) for name in numerical}
    harmonic = {}
    for name in numerical:
        amplitude, phase = harmonic_error(x, numerical[name], exact[name], bases[name], p["wave_number"])
        harmonic[name] = {"relative_amplitude_error": amplitude, "phase_error_radian": phase}
    drift = {name: float(abs(np.mean(numerical[name]) - bases[name])) for name in numerical}
    minimum = {name: float(np.min(values)) for name, values in numerical.items()}
    finite = all(np.all(np.isfinite(values)) for values in numerical.values())
    tolerances = {
        "maximum_relative_perturbation_l2": 0.15,
        "maximum_relative_harmonic_amplitude_error": 0.12,
        "maximum_harmonic_phase_error_radian": 0.10,
        "maximum_mean_drift": 5e-3,
    }
    passed = (
        finite
        and max(l2.values()) <= tolerances["maximum_relative_perturbation_l2"]
        and max(value["relative_amplitude_error"] for value in harmonic.values()) <= tolerances["maximum_relative_harmonic_amplitude_error"]
        and max(value["phase_error_radian"] for value in harmonic.values()) <= tolerances["maximum_harmonic_phase_error_radian"]
        and max(drift.values()) <= tolerances["maximum_mean_drift"]
        and minimum["rho"] > 0
        and minimum["rhoP"] > 0
        and minimum["p"] > 0
    )
    metrics = {
        "case": "dustyWave",
        "final_time_s": time,
        "metrics": {
            "relative_perturbation_l2": l2,
            "harmonic": harmonic,
            "mean_drift": drift,
            "minimum_values": minimum,
            "all_values_finite": finite,
        },
        "tolerances": tolerances,
        "passed": passed,
    }
    (ROOT / "dusty_wave_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    names = list(numerical)
    with (ROOT / "dusty_wave.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m"] + [f"{name}_exact" for name in names] + [f"{name}_numerical" for name in names])
        for index, coordinate in enumerate(x):
            writer.writerow([coordinate] + [exact[name][index] for name in names] + [numerical[name][index] for name in names])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "axes.linewidth": 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    figure, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    axes = list(axes.flat)
    labels = {
        "rho": r"$\rho_g-\rho_{g,0}$",
        "rhoP": r"$\rho_p-\rho_{p,0}$",
        "U": r"$U_g$",
        "Us": r"$U_p$",
        "p": r"$p_g-p_{g,0}$",
    }
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#DD8452"]
    markers = ["o", "s", "^", "D", "v"]
    for axis, name, color, marker in zip(axes, names, colors, markers):
        axis.plot(x, exact[name] - bases[name], color="black", linewidth=2.4, label="Analytical solution")
        axis.plot(x, numerical[name] - bases[name], color=color, linewidth=1.5, marker=marker, markerfacecolor="white", markeredgewidth=0.9, markersize=4.8, markevery=8, label="GPU-GKS-GKP")
        axis.set_xlabel(r"Axial position $x$ ($\mathrm{m}$)")
        axis.set_ylabel(labels[name], fontsize=20)
        axis.set_xlim(begin, end)
        axis.tick_params(top=True, right=True, length=6, width=1)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, legend_labels, frameon=False, loc="center", fontsize=16)
    figure.savefig(ROOT / "dusty_wave.png", dpi=600, facecolor="white")
    plt.close(figure)
    print(json.dumps({"relative_perturbation_l2": l2, "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
