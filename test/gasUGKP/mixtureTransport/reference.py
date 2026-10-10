#!/usr/bin/env python3
"""Independent exact cell averages for a periodic passive-mixture solution.

Equal molar masses and calorics make rho, U, p, T and rhoE uniform. Both
species have the same constant diffusivity, so mass-corrected Fick diffusion
reduces exactly to dY/dt + U*dY/dx = D*d2Y/dx2 and Y_B = 1-Y_A.
"""
import argparse
import json
import math
from pathlib import Path
import re


def species_cell_average(left, right, time, *, length, velocity, diffusivity,
                         mean=0.5, amplitude=0.2):
    if right <= left or length <= 0 or diffusivity < 0 or time < 0:
        raise ValueError("invalid cell, period, diffusivity or time")
    k = 2.0*math.pi/length
    phase_left = k*(left-velocity*time)
    phase_right = k*(right-velocity*time)
    sine_average = (math.cos(phase_left)-math.cos(phase_right))/(k*(right-left))
    return mean + amplitude*math.exp(-diffusivity*k*k*time)*sine_average


def read_scalar_field(path, count):
    text = Path(path).read_text()
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        return [float(uniform.group(1))]*count
    array = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not array:
        raise ValueError(f"No ASCII scalar internalField in {path}")
    values = [float(value) for value in array.group(2).split()]
    if int(array.group(1)) != count or len(values) != count:
        raise ValueError(f"Wrong cell count in {path}")
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"Nonfinite field in {path}")
    return values


def compare_case(case, time, directory=None):
    case = Path(case)
    params = json.loads((case / "case_parameters.json").read_text())
    n = params["cells"]
    directory = case / (directory if directory is not None else format(time, ".12g"))
    a = read_scalar_field(directory / "Y_A", n)
    b = read_scalar_field(directory / "Y_B", n)
    dx = params["length"]/n
    exact = [species_cell_average(i*dx, (i+1)*dx, time,
        **{key: params[key] for key in ("length", "velocity", "diffusivity", "mean", "amplitude")}) for i in range(n)]
    errors = [max(abs(x-y), abs(z-(1-y))) for x, z, y in zip(a,b,exact)]
    bulk = {name: read_scalar_field(directory / name, n) for name in ("rho", "p", "T", "rhoE")}
    targets = {"rho": params["rho"], "p": params["pressure"], "T": params["temperature"], "rhoE": params["rhoE"]}
    relative_bulk = max(abs(value-targets[name])/max(abs(targets[name]), 1e-300)
                        for name, values in bulk.items() for value in values)
    return {"time": time, "cells": n,
            "species_l1": sum(errors)/n, "species_linf": max(errors),
            "max_species_sum_error": max(abs(x+y-1) for x,y in zip(a,b)),
            "species_mass_mean_error": max(abs(sum(a)/n-params["mean"]),
                abs(sum(b)/n-(1-params["mean"]))),
            "max_relative_bulk_error": relative_bulk,
            "minimum_mass_fraction": min(a+b), "maximum_mass_fraction": max(a+b)}


def verify_native_run(log_path):
    text = Path(log_path).read_text()
    fields = ("api", "Ns", "mode", "speciesOrderHash", "thermoHash", "mechanismHash")
    pattern = r"Shared gas model:\s+" + r"\s+".join(name + r"=(\d+)" for name in fields)
    matches = re.findall(pattern, text)
    expected = dict(zip(fields, (1, 2, 1, 12445161613445958710, 17390867132586628798, 0)))
    if len(matches) != 1 or dict(zip(fields, map(int, matches[0]))) != expected:
        raise ValueError("native result requires the configured two-species mixture binary identity")
    if not re.search(r"^End\s*$", text, re.M):
        raise ValueError("native gasUGKP did not report successful interval completion")
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--time", type=float)
    parser.add_argument("--species-tolerance", type=float, default=0.012)
    parser.add_argument("--conservation-tolerance", type=float, default=2e-10)
    args = parser.parse_args()
    params = json.loads((args.case / "case_parameters.json").read_text())
    identity = verify_native_run(args.case / "log.gasUGKP")
    metrics = compare_case(args.case, params["end_time"] if args.time is None else args.time)
    metrics["backend_identity"] = identity
    print(json.dumps(metrics, sort_keys=True, indent=2))
    good = (metrics["species_linf"] <= args.species_tolerance
            and metrics["max_species_sum_error"] <= args.conservation_tolerance
            and metrics["species_mass_mean_error"] <= args.conservation_tolerance
            and metrics["max_relative_bulk_error"] <= args.conservation_tolerance
            and metrics["minimum_mass_fraction"] >= -args.conservation_tolerance
            and metrics["maximum_mass_fraction"] <= 1+args.conservation_tolerance)
    raise SystemExit(0 if good else 1)


if __name__ == "__main__":
    main()
