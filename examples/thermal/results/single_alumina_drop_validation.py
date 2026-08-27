from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


AXIAL_NODE_COUNT = 8
RADIAL_NODE_COUNT = 8
COLD_2D_NODE_COUNT = AXIAL_NODE_COUNT * RADIAL_NODE_COUNT
POSTPROCESS_END_MS = 10.0
ALUMINA_PROPERTY_TEMPERATURE_MIN_K = 2327.0
ALUMINA_PROPERTY_TEMPERATURE_MAX_K = 4000.0
ALUMINA_MOLAR_MASS_KG_MOL = 0.101961276
ALUMINA_CP_AT_MELTING_MOLAR_J_MOL_K = 153.5
ALUMINA_CP_MOLAR_SLOPE_J_MOL_K2 = 3.1e-3


@dataclass(frozen=True)
class ColdWallThermalProperties:
    melting_temperature_k: float
    mushy_range_k: float
    latent_heat_j_kg: float
    solid_specific_heat_j_kg_k: float


@dataclass(frozen=True)
class ParticleState:
    temperature_k: float
    contact_age_s: float
    diameter_m: float
    original_id: int
    wall_state: int
    damage_or_deposition_area_m2: float
    contact_duration_s: float
    maximum_contact_area_m2: float
    peak_time_fraction: float
    cold_frozen_area_m2: float
    cold_contact_age_s: float
    cold_node_specific_enthalpies_j_kg: tuple[float, ...]
    cold_2d_frozen_area_m2: float
    cold_2d_ring_contact_ages_s: tuple[float, ...]
    cold_2d_node_specific_enthalpies_j_kg: tuple[float, ...]


def _dictionary_scalar(path: Path, name: str):
    import re

    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(name)}\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*;",
        text,
    )
    if match is None:
        raise ValueError(f"{path} does not define scalar {name}")
    return float(match.group(1))


def read_cold_wall_thermal_properties(case_dir: Path):
    properties = case_dir / "constant" / "particleProperties"
    return ColdWallThermalProperties(
        melting_temperature_k=_dictionary_scalar(properties, "meltingTemperature"),
        mushy_range_k=_dictionary_scalar(properties, "mushyRange"),
        latent_heat_j_kg=_dictionary_scalar(properties, "latentHeat"),
        solid_specific_heat_j_kg_k=_dictionary_scalar(
            properties, "solidSpecificHeat"
        ),
    )


def alumina_liquid_specific_heat_j_kg_k(temperature_k: float):
    bounded_temperature_k = min(
        max(temperature_k, ALUMINA_PROPERTY_TEMPERATURE_MIN_K),
        ALUMINA_PROPERTY_TEMPERATURE_MAX_K,
    )
    return (
        ALUMINA_CP_AT_MELTING_MOLAR_J_MOL_K
        + ALUMINA_CP_MOLAR_SLOPE_J_MOL_K2
        * (bounded_temperature_k - ALUMINA_PROPERTY_TEMPERATURE_MIN_K)
    ) / ALUMINA_MOLAR_MASS_KG_MOL


@lru_cache(maxsize=None)
def cold_wall_mushy_specific_heat_j_kg_k(
    properties: ColdWallThermalProperties,
):
    solid_specific_heat = float(properties.solid_specific_heat_j_kg_k)
    liquid_specific_heat = alumina_liquid_specific_heat_j_kg_k(
        float(properties.melting_temperature_k)
    )
    latent_heat = float(properties.latent_heat_j_kg)
    mushy_range = float(properties.mushy_range_k)
    return 0.5 * (
        solid_specific_heat + liquid_specific_heat
    ) + latent_heat/mushy_range


def cold_wall_specific_enthalpy_j_kg(
    temperature_k: float,
    properties: ColdWallThermalProperties,
):
    solidus_k = properties.melting_temperature_k - 0.5 * properties.mushy_range_k
    liquidus_k = properties.melting_temperature_k + 0.5 * properties.mushy_range_k
    h_solidus = properties.solid_specific_heat_j_kg_k * solidus_k
    if temperature_k <= solidus_k:
        return properties.solid_specific_heat_j_kg_k * temperature_k
    mushy_cp = cold_wall_mushy_specific_heat_j_kg_k(properties)
    if temperature_k <= liquidus_k:
        return h_solidus + mushy_cp * (temperature_k - solidus_k)
    h_liquidus = h_solidus + mushy_cp * properties.mushy_range_k
    cp0 = alumina_liquid_specific_heat_j_kg_k(liquidus_k)
    cp1 = alumina_liquid_specific_heat_j_kg_k(temperature_k)
    return h_liquidus + 0.5 * (cp0 + cp1) * (temperature_k - liquidus_k)


def cold_wall_temperature_from_specific_enthalpy_k(
    specific_enthalpy_j_kg: float,
    properties: ColdWallThermalProperties,
):
    if not math.isfinite(specific_enthalpy_j_kg) or specific_enthalpy_j_kg < 0.0:
        raise ValueError("cold-wall node enthalpy must be finite and nonnegative")
    solidus_k = properties.melting_temperature_k - 0.5 * properties.mushy_range_k
    liquidus_k = properties.melting_temperature_k + 0.5 * properties.mushy_range_k
    h_solidus = properties.solid_specific_heat_j_kg_k * solidus_k
    if specific_enthalpy_j_kg <= h_solidus:
        return specific_enthalpy_j_kg / properties.solid_specific_heat_j_kg_k
    mushy_cp = cold_wall_mushy_specific_heat_j_kg_k(properties)
    h_liquidus = h_solidus + mushy_cp * properties.mushy_range_k
    if specific_enthalpy_j_kg <= h_liquidus:
        return solidus_k + (specific_enthalpy_j_kg - h_solidus) / mushy_cp
    lower_k = liquidus_k
    upper_k = liquidus_k + 1000.0
    while (
        cold_wall_specific_enthalpy_j_kg(upper_k, properties)
        < specific_enthalpy_j_kg
        and upper_k < 20000.0
    ):
        upper_k *= 1.5
    for _ in range(40):
        middle_k = 0.5 * (lower_k + upper_k)
        if (
            cold_wall_specific_enthalpy_j_kg(middle_k, properties)
            < specific_enthalpy_j_kg
        ):
            lower_k = middle_k
        else:
            upper_k = middle_k
    return 0.5 * (lower_k + upper_k)


def temperature_observables(
    state: ParticleState,
    properties: ColdWallThermalProperties | None,
):
    enthalpies_2d = state.cold_2d_node_specific_enthalpies_j_kg
    profile_2d_active = (
        len(enthalpies_2d) == COLD_2D_NODE_COUNT
        and all(math.isfinite(value) and value > 0.0 for value in enthalpies_2d)
    )
    enthalpies_1d = state.cold_node_specific_enthalpies_j_kg
    profile_1d_active = (
        len(enthalpies_1d) == AXIAL_NODE_COUNT
        and all(math.isfinite(value) and value > 0.0 for value in enthalpies_1d)
    )
    if profile_2d_active:
        if properties is None:
            raise ValueError(
                "active cold-wall temperature profile lacks thermal properties"
            )
        node_temperatures_k = tuple(
            cold_wall_temperature_from_specific_enthalpy_k(value, properties)
            for value in enthalpies_2d
        )
        wall_contact_temperature_k = sum(
            node_temperatures_k[ring * AXIAL_NODE_COUNT]
            for ring in range(RADIAL_NODE_COUNT)
        ) / RADIAL_NODE_COUNT
        free_surface_temperature_k = sum(
            node_temperatures_k[
                ring * AXIAL_NODE_COUNT + AXIAL_NODE_COUNT - 1
            ]
            for ring in range(RADIAL_NODE_COUNT)
        ) / RADIAL_NODE_COUNT
        profile_dimension = 2
    elif profile_1d_active:
        if properties is None:
            raise ValueError(
                "active cold-wall temperature profile lacks thermal properties"
            )
        node_temperatures_k = tuple(
            cold_wall_temperature_from_specific_enthalpy_k(value, properties)
            for value in enthalpies_1d
        )
        wall_contact_temperature_k = node_temperatures_k[0]
        free_surface_temperature_k = node_temperatures_k[-1]
        profile_dimension = 1
    else:
        node_temperatures_k = (state.temperature_k,) * AXIAL_NODE_COUNT
        wall_contact_temperature_k = state.temperature_k
        free_surface_temperature_k = state.temperature_k
        profile_dimension = 0
    return {
        "enthalpy_equivalent_temperature_k": state.temperature_k,
        "wall_contact_layer_temperature_k": wall_contact_temperature_k,
        "axial_node_mean_temperature_k": sum(node_temperatures_k)
        / len(node_temperatures_k),
        "free_surface_temperature_k": free_surface_temperature_k,
        "cold_profile_active": profile_dimension,
    }


def _read_values(stream, code: str, count: int):
    size = struct.calcsize("<" + code) * count
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("truncated particle restart payload")
    return struct.unpack("<" + str(count) + code, payload)


def _parse_schema3(path: Path, header: str):
    tokens = header.split()
    if len(tokens) != 2:
        raise ValueError("invalid schema-3 particle restart header")
    count = int(tokens[1])
    lines = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    if len(lines) != count:
        raise ValueError("schema-3 particle count does not match payload")
    states = []
    for values in lines:
        if len(values) != 16:
            raise ValueError("schema-3 particle record must contain 16 values")
        states.append(
            ParticleState(
                temperature_k=float(values[6]),
                contact_age_s=float(values[7]),
                diameter_m=float(values[8]),
                original_id=int(values[12]),
                wall_state=int(values[13]),
                damage_or_deposition_area_m2=float(values[15]),
                contact_duration_s=0.0,
                maximum_contact_area_m2=0.0,
                peak_time_fraction=0.0,
                cold_frozen_area_m2=0.0,
                cold_contact_age_s=0.0,
                cold_node_specific_enthalpies_j_kg=(),
                cold_2d_frozen_area_m2=0.0,
                cold_2d_ring_contact_ages_s=(),
                cold_2d_node_specific_enthalpies_j_kg=(),
            )
        )
    return states


def parse_particle_restart(path: Path):
    with path.open("rb") as stream:
        raw_header = stream.readline()
        try:
            header = raw_header.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("particle restart header is not ASCII") from error
        if header.startswith("UGKP_PARTICLES_SCHEMA3"):
            return _parse_schema3(path, header)
        tokens = header.split()
        if len(tokens) != 3 or tokens[0] not in (
            "UGKP_FSH_PARTICLES_SCHEMA2_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA3_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA4_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA5_BIN",
        ):
            raise ValueError("unsupported particle restart format: " + header)
        has_peak_fraction = tokens[0] in (
            "UGKP_FSH_PARTICLES_SCHEMA3_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA4_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA5_BIN",
        )
        has_cold_profile = tokens[0] in (
            "UGKP_FSH_PARTICLES_SCHEMA4_BIN",
            "UGKP_FSH_PARTICLES_SCHEMA5_BIN",
        )
        has_cold_2d_profile = tokens[0] == "UGKP_FSH_PARTICLES_SCHEMA5_BIN"
        count = int(tokens[1])
        maximum_chunk = int(tokens[2])
        if count < 0 or maximum_chunk <= 0:
            raise ValueError("invalid particle restart count or chunk bound")
        formats = (
            ("px", "d"),
            ("py", "d"),
            ("pz", "d"),
            ("pux", "d"),
            ("puy", "d"),
            ("puz", "d"),
            ("temperature", "d"),
            ("contact_age", "d"),
            ("diameter", "d"),
            ("mass", "d"),
            ("cell", "i"),
            ("status", "i"),
            ("rng", "Q"),
            ("original_id", "Q"),
            ("wall_state", "B"),
            ("wall_face", "i"),
            ("area", "f"),
            ("duration", "f"),
            ("maximum_area", "f"),
        )
        if has_peak_fraction:
            formats += (("peak_fraction", "f"),)
        arrays = {name: [] for name, _ in formats}
        arrays["cold_node_specific_enthalpy"] = []
        arrays["cold_frozen_area"] = []
        arrays["cold_contact_age"] = []
        arrays["cold_2d_node_specific_enthalpy"] = []
        arrays["cold_2d_ring_contact_age"] = []
        arrays["cold_2d_frozen_area"] = []
        loaded = 0
        while loaded < count:
            chunk = _read_values(stream, "I", 1)[0]
            if chunk <= 0 or chunk > maximum_chunk or loaded + chunk > count:
                raise ValueError("invalid particle restart chunk size")
            for name, code in formats:
                arrays[name].extend(_read_values(stream, code, chunk))
            if has_cold_profile:
                arrays["cold_node_specific_enthalpy"].extend(
                    _read_values(stream, "f", chunk * AXIAL_NODE_COUNT)
                )
                _read_values(stream, "f", chunk * 8)
                arrays["cold_frozen_area"].extend(
                    _read_values(stream, "f", chunk)
                )
                arrays["cold_contact_age"].extend(
                    _read_values(stream, "f", chunk)
                )
            else:
                arrays["cold_node_specific_enthalpy"].extend(
                    [0.0] * (chunk * AXIAL_NODE_COUNT)
                )
                arrays["cold_frozen_area"].extend([0.0] * chunk)
                arrays["cold_contact_age"].extend([0.0] * chunk)
            if has_cold_2d_profile:
                arrays["cold_2d_node_specific_enthalpy"].extend(
                    _read_values(stream, "f", chunk * COLD_2D_NODE_COUNT)
                )
                arrays["cold_2d_ring_contact_age"].extend(
                    _read_values(stream, "f", chunk * RADIAL_NODE_COUNT)
                )
                arrays["cold_2d_frozen_area"].extend(
                    _read_values(stream, "f", chunk)
                )
            else:
                arrays["cold_2d_node_specific_enthalpy"].extend(
                    [0.0] * (chunk * COLD_2D_NODE_COUNT)
                )
                arrays["cold_2d_ring_contact_age"].extend(
                    [0.0] * (chunk * RADIAL_NODE_COUNT)
                )
                arrays["cold_2d_frozen_area"].extend([0.0] * chunk)
            loaded += chunk
        if stream.read(1):
            raise ValueError("trailing data in particle restart")
    return [
        ParticleState(
            temperature_k=arrays["temperature"][i],
            contact_age_s=arrays["contact_age"][i],
            diameter_m=arrays["diameter"][i],
            original_id=arrays["original_id"][i],
            wall_state=arrays["wall_state"][i],
            damage_or_deposition_area_m2=arrays["area"][i],
            contact_duration_s=arrays["duration"][i],
            maximum_contact_area_m2=arrays["maximum_area"][i],
            peak_time_fraction=(
                arrays["peak_fraction"][i]
                if has_peak_fraction
                else (0.5 if arrays["wall_state"][i] in (2, 3) else 0.0)
            ),
            cold_frozen_area_m2=arrays["cold_frozen_area"][i],
            cold_contact_age_s=arrays["cold_contact_age"][i],
            cold_node_specific_enthalpies_j_kg=tuple(
                arrays["cold_node_specific_enthalpy"][
                    i * AXIAL_NODE_COUNT : (i + 1) * AXIAL_NODE_COUNT
                ]
            ),
            cold_2d_frozen_area_m2=arrays["cold_2d_frozen_area"][i],
            cold_2d_ring_contact_ages_s=tuple(
                arrays["cold_2d_ring_contact_age"][
                    i * RADIAL_NODE_COUNT : (i + 1) * RADIAL_NODE_COUNT
                ]
            ),
            cold_2d_node_specific_enthalpies_j_kg=tuple(
                arrays["cold_2d_node_specific_enthalpy"][
                    i * COLD_2D_NODE_COUNT : (i + 1) * COLD_2D_NODE_COUNT
                ]
            ),
        )
        for i in range(count)
        if arrays["status"][i] != 0
    ]


def effective_contact_area(state: ParticleState):
    if state.wall_state == 1:
        return max(state.damage_or_deposition_area_m2, 0.0)
    if state.wall_state in (2, 3):
        if (
            state.contact_duration_s <= 0.0
            or state.maximum_contact_area_m2 <= 0.0
            or not 0.0 < state.peak_time_fraction < 1.0
        ):
            raise ValueError("transient contact state lacks finite-contact metadata")
        normalized_age = min(max(state.contact_age_s / state.contact_duration_s, 0.0), 1.0)
        if normalized_age <= state.peak_time_fraction:
            coordinate = normalized_age / state.peak_time_fraction
            normalized_area = coordinate * (2.0 - coordinate)
        else:
            coordinate = (
                (normalized_age - state.peak_time_fraction)
                / (1.0 - state.peak_time_fraction)
            )
            normalized_area = math.cos(0.5 * math.pi * coordinate) ** 2
        kinematic = state.maximum_contact_area_m2 * normalized_area
        intrinsic = max(
            kinematic,
            state.cold_frozen_area_m2,
            state.cold_2d_frozen_area_m2,
        )
        return max(intrinsic - state.damage_or_deposition_area_m2, 0.0)
    return 0.0


def spreading_factor(state: ParticleState):
    area = effective_contact_area(state)
    if area <= 0.0:
        return 0.0
    return math.sqrt(4.0 * area / math.pi) / state.diameter_m


def collect_model_rows(case_dir: Path):
    samples = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            absolute_time = float(entry.name)
        except ValueError:
            continue
        restart = entry / "gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        states = parse_particle_restart(restart)
        if not states:
            continue
        state = min(states, key=lambda item: item.original_id)
        samples.append((absolute_time, state))
    samples.sort(key=lambda item: item[0])
    has_active_cold_profile = any(
        (
            len(state.cold_node_specific_enthalpies_j_kg) == AXIAL_NODE_COUNT
            and all(
                math.isfinite(value) and value > 0.0
                for value in state.cold_node_specific_enthalpies_j_kg
            )
        )
        or (
            len(state.cold_2d_node_specific_enthalpies_j_kg)
            == COLD_2D_NODE_COUNT
            and all(
                math.isfinite(value) and value > 0.0
                for value in state.cold_2d_node_specific_enthalpies_j_kg
            )
        )
        for _, state in samples
    )
    thermal_properties = (
        read_cold_wall_thermal_properties(case_dir)
        if has_active_cold_profile
        else None
    )
    contact_samples = [item for item in samples if item[1].wall_state != 0]
    if not contact_samples:
        raise RuntimeError("no written wall-contact state was found")
    first_contact_time, first_contact_state = contact_samples[0]
    impact_time = first_contact_time - first_contact_state.contact_age_s
    rows = []
    mobile_before_impact = [item for item in samples if item[0] <= impact_time]
    if mobile_before_impact:
        initial_state = mobile_before_impact[-1][1]
        temperatures = temperature_observables(initial_state, thermal_properties)
        rows.append(
            {
                "time_abs_s": impact_time,
                "time_after_impact_ms": 0.0,
                "model_enthalpy_equivalent_temperature_K": temperatures[
                    "enthalpy_equivalent_temperature_k"
                ],
                "model_enthalpy_equivalent_temperature_C": temperatures[
                    "enthalpy_equivalent_temperature_k"
                ] - 273.15,
                "model_wall_contact_layer_temperature_K": temperatures[
                    "wall_contact_layer_temperature_k"
                ],
                "model_wall_contact_layer_temperature_C": temperatures[
                    "wall_contact_layer_temperature_k"
                ] - 273.15,
                "model_axial_node_mean_temperature_K": temperatures[
                    "axial_node_mean_temperature_k"
                ],
                "model_axial_node_mean_temperature_C": temperatures[
                    "axial_node_mean_temperature_k"
                ] - 273.15,
                "model_free_surface_temperature_K": temperatures[
                    "free_surface_temperature_k"
                ],
                "model_free_surface_temperature_C": temperatures[
                    "free_surface_temperature_k"
                ] - 273.15,
                "cold_profile_active": temperatures["cold_profile_active"],
                "model_beta": 0.0,
                "wall_state": first_contact_state.wall_state,
                "contact_area_m2": 0.0,
                "cold_frozen_area_m2": 0.0,
                "cold_contact_age_s": 0.0,
            }
        )
    for absolute_time, state in samples:
        relative_ms = (absolute_time - impact_time) * 1000.0
        if relative_ms <= 1.0e-9 or relative_ms > 14.000001:
            continue
        area = effective_contact_area(state)
        temperatures = temperature_observables(state, thermal_properties)
        rows.append(
            {
                "time_abs_s": absolute_time,
                "time_after_impact_ms": max(relative_ms, 0.0),
                "model_enthalpy_equivalent_temperature_K": temperatures[
                    "enthalpy_equivalent_temperature_k"
                ],
                "model_enthalpy_equivalent_temperature_C": temperatures[
                    "enthalpy_equivalent_temperature_k"
                ] - 273.15,
                "model_wall_contact_layer_temperature_K": temperatures[
                    "wall_contact_layer_temperature_k"
                ],
                "model_wall_contact_layer_temperature_C": temperatures[
                    "wall_contact_layer_temperature_k"
                ] - 273.15,
                "model_axial_node_mean_temperature_K": temperatures[
                    "axial_node_mean_temperature_k"
                ],
                "model_axial_node_mean_temperature_C": temperatures[
                    "axial_node_mean_temperature_k"
                ] - 273.15,
                "model_free_surface_temperature_K": temperatures[
                    "free_surface_temperature_k"
                ],
                "model_free_surface_temperature_C": temperatures[
                    "free_surface_temperature_k"
                ] - 273.15,
                "cold_profile_active": temperatures["cold_profile_active"],
                "model_beta": spreading_factor(state),
                "wall_state": state.wall_state,
                "contact_area_m2": area,
                "cold_frozen_area_m2": max(
                    state.cold_frozen_area_m2,
                    state.cold_2d_frozen_area_m2,
                ),
                "cold_contact_age_s": max(
                    (state.cold_contact_age_s,)
                    + state.cold_2d_ring_contact_ages_s
                ),
            }
        )
    return impact_time, rows
