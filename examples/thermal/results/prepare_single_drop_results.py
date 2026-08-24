from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path


RESULTS = Path(__file__).resolve().parent
CASE = RESULTS.parent / "singleAluminaDrop"
POSTPROCESS_END_MS = 10.0
MODULE_PATH = RESULTS / "single_alumina_drop_validation.py"
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
            "enthalpy_equivalent_temperature_C": float(
                row["model_enthalpy_equivalent_temperature_C"]
            ),
            "enthalpy_equivalent_temperature_K": float(
                row["model_enthalpy_equivalent_temperature_K"]
            ),
            "wall_contact_layer_temperature_C": float(
                row["model_wall_contact_layer_temperature_C"]
            ),
            "wall_contact_layer_temperature_K": float(
                row["model_wall_contact_layer_temperature_K"]
            ),
            "axial_node_mean_temperature_C": float(
                row["model_axial_node_mean_temperature_C"]
            ),
            "axial_node_mean_temperature_K": float(
                row["model_axial_node_mean_temperature_K"]
            ),
            "free_surface_temperature_C": float(
                row["model_free_surface_temperature_C"]
            ),
            "free_surface_temperature_K": float(
                row["model_free_surface_temperature_K"]
            ),
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


def write_columns(path: Path, fieldnames, datasets):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(max(len(rows) for rows in datasets)):
            row = {}
            for rows in datasets:
                if index < len(rows):
                    row.update(rows[index])
            writer.writerow(row)


def interpolate(model_rows, time_ms, field):
    if time_ms <= model_rows[0]["time_ms"]:
        return model_rows[0][field]
    if time_ms >= model_rows[-1]["time_ms"]:
        return model_rows[-1][field]
    for left, right in zip(model_rows, model_rows[1:]):
        if left["time_ms"] <= time_ms <= right["time_ms"]:
            fraction = (time_ms - left["time_ms"]) / (
                right["time_ms"] - left["time_ms"]
            )
            return left[field] + fraction * (right[field] - left[field])
    raise RuntimeError("model interpolation interval was not found")


def metrics(experiment, model, experiment_field, model_field):
    errors = [
        interpolate(model, float(row["time_ms"]), model_field)
        - float(row[experiment_field])
        for row in experiment
    ]
    return {
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "maximum_absolute_error": max(abs(error) for error in errors),
    }


def main():
    hot_impact, hot = normalize_model(CASE / "hotWall")
    cold_impact, cold = normalize_model(CASE / "coldWall")
    temperature_experiment = read_csv(
        RESULTS / "data" / "experiment_surface_temperature.csv"
    )
    beta_experiment = read_csv(RESULTS / "data" / "experiment_beta.csv")
    temperature_experiment = [
        row
        for row in temperature_experiment
        if float(row["time_ms"]) <= POSTPROCESS_END_MS + 1.0e-9
    ]
    beta_experiment = [
        row
        for row in beta_experiment
        if float(row["time_ms"]) <= POSTPROCESS_END_MS + 1.0e-9
    ]

    temperature_experiment_rows = [
        {
            "experiment_time_ms": float(row["time_ms"]),
            "experiment_surface_temperature_C": float(row["T_surface_C"]),
        }
        for row in temperature_experiment
    ]

    def hot_temperature_rows(model):
        return [
            {
                "hot_time_ms": row["time_ms"],
                "hot_lumped_temperature_C": row[
                    "enthalpy_equivalent_temperature_C"
                ],
                "hot_lumped_temperature_K": row[
                    "enthalpy_equivalent_temperature_K"
                ],
                "hot_wall_state": row["wall_state"],
                "hot_contact_area_m2": row["contact_area_m2"],
                "hot_frozen_area_m2": row["frozen_area_m2"],
                "hot_solidification_contact_age_s": row[
                    "solidification_contact_age_s"
                ],
            }
            for row in model
        ]

    def cold_temperature_rows(prefix, model):
        return [
            {
                f"{prefix}_time_ms": row["time_ms"],
                f"{prefix}_wall_contact_layer_temperature_C": row[
                    "wall_contact_layer_temperature_C"
                ],
                f"{prefix}_wall_contact_layer_temperature_K": row[
                    "wall_contact_layer_temperature_K"
                ],
                f"{prefix}_enthalpy_equivalent_temperature_C": row[
                    "enthalpy_equivalent_temperature_C"
                ],
                f"{prefix}_enthalpy_equivalent_temperature_K": row[
                    "enthalpy_equivalent_temperature_K"
                ],
                f"{prefix}_axial_node_mean_temperature_C": row[
                    "axial_node_mean_temperature_C"
                ],
                f"{prefix}_axial_node_mean_temperature_K": row[
                    "axial_node_mean_temperature_K"
                ],
                f"{prefix}_free_surface_temperature_C": row[
                    "free_surface_temperature_C"
                ],
                f"{prefix}_free_surface_temperature_K": row[
                    "free_surface_temperature_K"
                ],
                f"{prefix}_profile_active": row["cold_profile_active"],
                f"{prefix}_wall_state": row["wall_state"],
                f"{prefix}_contact_area_m2": row["contact_area_m2"],
                f"{prefix}_frozen_area_m2": row["frozen_area_m2"],
                f"{prefix}_solidification_contact_age_s": row[
                    "solidification_contact_age_s"
                ],
            }
            for row in model
        ]

    write_columns(
        RESULTS / "single_drop_temperature_comparison.csv",
        (
            "experiment_time_ms",
            "experiment_surface_temperature_C",
            "hot_time_ms",
            "hot_lumped_temperature_C",
            "hot_lumped_temperature_K",
            "hot_wall_state",
            "hot_contact_area_m2",
            "hot_frozen_area_m2",
            "hot_solidification_contact_age_s",
            "cold_time_ms",
            "cold_wall_contact_layer_temperature_C",
            "cold_wall_contact_layer_temperature_K",
            "cold_enthalpy_equivalent_temperature_C",
            "cold_enthalpy_equivalent_temperature_K",
            "cold_axial_node_mean_temperature_C",
            "cold_axial_node_mean_temperature_K",
            "cold_free_surface_temperature_C",
            "cold_free_surface_temperature_K",
            "cold_profile_active",
            "cold_wall_state",
            "cold_contact_area_m2",
            "cold_frozen_area_m2",
            "cold_solidification_contact_age_s",
        ),
        (
            temperature_experiment_rows,
            hot_temperature_rows(hot),
            cold_temperature_rows("cold", cold),
        ),
    )

    beta_experiment_rows = [
        {
            "experiment_time_ms": float(row["time_ms"]),
            "experiment_beta": float(row["beta"]),
        }
        for row in beta_experiment
    ]

    def spreading_rows(prefix, model):
        return [
            {
                f"{prefix}_time_ms": row["time_ms"],
                f"{prefix}_beta": row["beta"],
                f"{prefix}_wall_state": row["wall_state"],
                f"{prefix}_contact_area_m2": row["contact_area_m2"],
                f"{prefix}_frozen_area_m2": row["frozen_area_m2"],
            }
            for row in model
        ]

    write_columns(
        RESULTS / "single_drop_spreading_comparison.csv",
        (
            "experiment_time_ms",
            "experiment_beta",
            "hot_time_ms",
            "hot_beta",
            "hot_wall_state",
            "hot_contact_area_m2",
            "hot_frozen_area_m2",
            "cold_time_ms",
            "cold_beta",
            "cold_wall_state",
            "cold_contact_area_m2",
            "cold_frozen_area_m2",
        ),
        (
            beta_experiment_rows,
            spreading_rows("hot", hot),
            spreading_rows("cold", cold),
        ),
    )

    metric_rows = []
    metric_definitions = (
        (
            "hotWall",
            "lumped_temperature",
            "degC",
            hot,
            temperature_experiment,
            "T_surface_C",
            "enthalpy_equivalent_temperature_C",
        ),
        (
            "coldWall1D",
            "free_surface_temperature",
            "degC",
            cold,
            temperature_experiment,
            "T_surface_C",
            "free_surface_temperature_C",
        ),
        (
            "hotWall",
            "spreading_factor",
            "1",
            hot,
            beta_experiment,
            "beta",
            "beta",
        ),
        (
            "coldWall1D",
            "spreading_factor",
            "1",
            cold,
            beta_experiment,
            "beta",
            "beta",
        ),
    )
    for (
        model_name,
        quantity,
        unit,
        model,
        experiment,
        experiment_field,
        model_field,
    ) in metric_definitions:
        for metric, value in metrics(
            experiment, model, experiment_field, model_field
        ).items():
            metric_rows.append(
                {
                    "model": model_name,
                    "quantity": quantity,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                }
            )
    with (RESULTS / "single_drop_comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("model", "quantity", "metric", "value", "unit")
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"hot_impact_time_s={hot_impact:.12g} hot_records={len(hot)}")
    print(f"cold_impact_time_s={cold_impact:.12g} cold_records={len(cold)}")


if __name__ == "__main__":
    main()
