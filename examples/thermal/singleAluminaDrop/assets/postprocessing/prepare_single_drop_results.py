from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CASE_ROOT.parent/"results"/CASE_ROOT.name/"coldWall"
DATA = RESULTS/"data"
CASE = CASE_ROOT/"coldWall"
POSTPROCESS_END_MS = 10.0
MODULE_PATH = Path(__file__).resolve().parent/"single_alumina_drop_validation.py"
ASSET_DATA = Path(__file__).resolve().parent/"data"
SPEC = importlib.util.spec_from_file_location("single_drop_post", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize_model(case_dir: Path):
    impact_time, raw_rows = MODULE.collect_model_rows(case_dir)
    rows = [
        {
            "time_ms": float(row["time_after_impact_ms"]),
            "enthalpy_equivalent_temperature_C": float(row["model_enthalpy_equivalent_temperature_C"]),
            "enthalpy_equivalent_temperature_K": float(row["model_enthalpy_equivalent_temperature_K"]),
            "wall_contact_layer_temperature_C": float(row["model_wall_contact_layer_temperature_C"]),
            "wall_contact_layer_temperature_K": float(row["model_wall_contact_layer_temperature_K"]),
            "axial_node_mean_temperature_C": float(row["model_axial_node_mean_temperature_C"]),
            "axial_node_mean_temperature_K": float(row["model_axial_node_mean_temperature_K"]),
            "free_surface_temperature_C": float(row["model_free_surface_temperature_C"]),
            "free_surface_temperature_K": float(row["model_free_surface_temperature_K"]),
            "cold_profile_active": int(row["cold_profile_active"]),
            "beta": float(row["model_beta"]),
            "wall_state": int(row["wall_state"]),
            "contact_area_m2": float(row["contact_area_m2"]),
            "frozen_area_m2": float(row["cold_frozen_area_m2"]),
            "solidification_contact_age_s": float(row["cold_contact_age_s"]),
        }
        for row in raw_rows
        if float(row["time_after_impact_ms"]) <= POSTPROCESS_END_MS + 1.0e-9
    ]
    return impact_time, rows


def interpolate(model_rows, time_ms, field):
    if time_ms <= model_rows[0]["time_ms"]:
        return model_rows[0][field]
    if time_ms >= model_rows[-1]["time_ms"]:
        return model_rows[-1][field]
    for left, right in zip(model_rows, model_rows[1:]):
        if left["time_ms"] <= time_ms <= right["time_ms"]:
            fraction = (time_ms - left["time_ms"])/(right["time_ms"] - left["time_ms"])
            return left[field] + fraction*(right[field] - left[field])
    raise RuntimeError("model interpolation interval was not found")


def metrics(experiment, model, experiment_field, model_field):
    errors = [
        interpolate(model, float(row["time_ms"]), model_field) - float(row[experiment_field])
        for row in experiment
    ]
    return {
        "rmse": math.sqrt(sum(value*value for value in errors)/len(errors)),
        "mae": sum(abs(value) for value in errors)/len(errors),
        "bias": sum(errors)/len(errors),
        "maximum_absolute_error": max(abs(value) for value in errors),
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    impact_time, cold = normalize_model(CASE)
    temperature_experiment = [
        row for row in read_csv(ASSET_DATA/"experiment_surface_temperature.csv")
        if float(row["time_ms"]) <= POSTPROCESS_END_MS + 1.0e-9
    ]
    beta_experiment = [
        row for row in read_csv(ASSET_DATA/"experiment_beta.csv")
        if float(row["time_ms"]) <= POSTPROCESS_END_MS + 1.0e-9
    ]

    with (DATA/"single_drop_temperature_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "record_type", "time_ms", "experiment_surface_temperature_C",
            "cold_wall_contact_layer_temperature_C", "cold_wall_contact_layer_temperature_K",
            "cold_enthalpy_equivalent_temperature_C", "cold_enthalpy_equivalent_temperature_K",
            "cold_axial_node_mean_temperature_C", "cold_axial_node_mean_temperature_K",
            "cold_free_surface_temperature_C", "cold_free_surface_temperature_K",
            "cold_profile_active", "cold_wall_state", "cold_contact_area_m2",
            "cold_frozen_area_m2", "cold_solidification_contact_age_s",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in temperature_experiment:
            writer.writerow({
                "record_type": "experiment",
                "time_ms": float(row["time_ms"]),
                "experiment_surface_temperature_C": float(row["T_surface_C"]),
            })
        for row in cold:
            writer.writerow({
                "record_type": "coldWall",
                "time_ms": row["time_ms"],
                "cold_wall_contact_layer_temperature_C": row["wall_contact_layer_temperature_C"],
                "cold_wall_contact_layer_temperature_K": row["wall_contact_layer_temperature_K"],
                "cold_enthalpy_equivalent_temperature_C": row["enthalpy_equivalent_temperature_C"],
                "cold_enthalpy_equivalent_temperature_K": row["enthalpy_equivalent_temperature_K"],
                "cold_axial_node_mean_temperature_C": row["axial_node_mean_temperature_C"],
                "cold_axial_node_mean_temperature_K": row["axial_node_mean_temperature_K"],
                "cold_free_surface_temperature_C": row["free_surface_temperature_C"],
                "cold_free_surface_temperature_K": row["free_surface_temperature_K"],
                "cold_profile_active": row["cold_profile_active"],
                "cold_wall_state": row["wall_state"],
                "cold_contact_area_m2": row["contact_area_m2"],
                "cold_frozen_area_m2": row["frozen_area_m2"],
                "cold_solidification_contact_age_s": row["solidification_contact_age_s"],
            })

    with (DATA/"single_drop_spreading_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "record_type", "time_ms", "experiment_beta", "cold_beta",
            "cold_wall_state", "cold_contact_area_m2", "cold_frozen_area_m2",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in beta_experiment:
            writer.writerow({
                "record_type": "experiment",
                "time_ms": float(row["time_ms"]),
                "experiment_beta": float(row["beta"]),
            })
        for row in cold:
            writer.writerow({
                "record_type": "coldWall",
                "time_ms": row["time_ms"],
                "cold_beta": row["beta"],
                "cold_wall_state": row["wall_state"],
                "cold_contact_area_m2": row["contact_area_m2"],
                "cold_frozen_area_m2": row["frozen_area_m2"],
            })

    metric_definitions = (
        ("free_surface_temperature", "degC", temperature_experiment, "T_surface_C", "free_surface_temperature_C"),
        ("spreading_factor", "1", beta_experiment, "beta", "beta"),
    )
    with (DATA/"single_drop_comparison_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "quantity", "metric", "value", "unit"))
        writer.writeheader()
        for quantity, unit, experiment, experiment_field, model_field in metric_definitions:
            for name, value in metrics(experiment, cold, experiment_field, model_field).items():
                writer.writerow({
                    "model": "coldWall",
                    "quantity": quantity,
                    "metric": name,
                    "value": value,
                    "unit": unit,
                })

    print(f"impact_time_s={impact_time:.12g} cold_records={len(cold)}")


if __name__ == "__main__":
    main()
