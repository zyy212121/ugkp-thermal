import importlib.util
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "examples"
    / "thermal"
    / "results"
    / "single_alumina_drop_validation.py"
)
SPEC = importlib.util.spec_from_file_location("single_drop_post", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_restart(
    path,
    age=0.25,
    duration=1.0,
    maximum_area=math.pi / 4.0,
    peak_fraction=0.25,
    wall_state=3,
):
    fields = (
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 2373.15),
        ("d", age),
        ("d", 1.0),
        ("d", 1.0),
        ("i", 0),
        ("i", 1),
        ("Q", 1),
        ("Q", 0),
        ("B", wall_state),
        ("i", 0),
        ("f", 0.0),
        ("f", duration),
        ("f", maximum_area),
        ("f", peak_fraction),
    )
    with path.open("wb") as stream:
        stream.write(b"UGKP_FSH_PARTICLES_SCHEMA3_BIN 1 262144\n")
        stream.write(struct.pack("<I", 1))
        for code, value in fields:
            stream.write(struct.pack("<" + code, value))


def write_particle_properties(case):
    constant = case / "constant"
    constant.mkdir(exist_ok=True)
    (constant / "particleProperties").write_text(
        "meltingTemperature 2327;\n"
        "mushyRange 20;\n"
        "latentHeat 1.16e6;\n"
        "solidSpecificHeat 1273;\n",
        encoding="utf-8",
    )


def write_schema4_restart(path, node_temperatures_k):
    properties = MODULE.ColdWallThermalProperties(2327.0, 20.0, 1.16e6, 1273.0)
    fields = (
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 1800.0),
        ("d", 0.001),
        ("d", 1.0),
        ("d", 1.0),
        ("i", 0),
        ("i", 1),
        ("Q", 1),
        ("Q", 0),
        ("B", 3),
        ("i", 0),
        ("f", 0.0),
        ("f", 0.01),
        ("f", math.pi / 4.0),
        ("f", 0.25),
    )
    with path.open("wb") as stream:
        stream.write(b"UGKP_FSH_PARTICLES_SCHEMA4_BIN 1 262144\n")
        stream.write(struct.pack("<I", 1))
        for code, value in fields:
            stream.write(struct.pack("<" + code, value))
        enthalpies = [
            MODULE.cold_wall_specific_enthalpy_j_kg(value, properties)
            for value in node_temperatures_k
        ]
        stream.write(struct.pack("<8f", *enthalpies))
        stream.write(struct.pack("<8f", *([0.0] * 8)))
        stream.write(struct.pack("<f", 0.0))
        stream.write(struct.pack("<f", 0.001))


def write_schema5_restart(path, node_temperatures_k):
    properties = MODULE.ColdWallThermalProperties(2327.0, 20.0, 1.16e6, 1273.0)
    fields = (
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 0.0),
        ("d", 1800.0),
        ("d", 0.001),
        ("d", 1.0),
        ("d", 1.0),
        ("i", 0),
        ("i", 1),
        ("Q", 1),
        ("Q", 0),
        ("B", 3),
        ("i", 0),
        ("f", 0.0),
        ("f", 0.01),
        ("f", math.pi / 4.0),
        ("f", 0.25),
    )
    with path.open("wb") as stream:
        stream.write(b"UGKP_FSH_PARTICLES_SCHEMA5_BIN 1 262144\n")
        stream.write(struct.pack("<I", 1))
        for code, value in fields:
            stream.write(struct.pack("<" + code, value))
        stream.write(struct.pack("<8f", *([0.0] * 8)))
        stream.write(struct.pack("<8f", *([0.0] * 8)))
        stream.write(struct.pack("<f", 0.0))
        stream.write(struct.pack("<f", 0.0))
        enthalpies = [
            MODULE.cold_wall_specific_enthalpy_j_kg(value, properties)
            for value in node_temperatures_k
        ]
        stream.write(struct.pack("<64f", *enthalpies))
        stream.write(struct.pack("<8f", *([0.001] * 8)))
        stream.write(struct.pack("<f", 0.2))


class PostprocessTests(unittest.TestCase):
    def test_schema3_and_beta_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            restart = Path(directory) / "gpuResidentStrictParticles.dat"
            write_restart(restart)
            state = MODULE.parse_particle_restart(restart)[0]
            expected_area = math.pi / 4.0
            self.assertAlmostEqual(MODULE.effective_contact_area(state), expected_area, places=6)
            self.assertAlmostEqual(MODULE.spreading_factor(state), 1.0, places=6)

    def test_impact_alignment_recovers_sub_write_contact_time(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            write_particle_properties(case)
            for time_name, age, wall_state in (
                ("0", 0.0, 0),
                ("0.00005", 0.00002, 3),
                ("0.00010", 0.00007, 3),
            ):
                time_dir = case / time_name
                time_dir.mkdir()
                write_restart(
                    time_dir / "gpuResidentStrictParticles.dat",
                    age=age,
                    duration=0.013846,
                    maximum_area=4.25e-4,
                    wall_state=wall_state,
                )
            impact_time, rows = MODULE.collect_model_rows(case)
            self.assertAlmostEqual(impact_time, 0.00003)
            self.assertAlmostEqual(rows[0]["time_after_impact_ms"], 0.0)
            self.assertAlmostEqual(rows[1]["time_after_impact_ms"], 0.02)
            self.assertAlmostEqual(rows[2]["time_after_impact_ms"], 0.07)

    def test_schema4_exposes_wall_contact_layer_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            restart = Path(directory) / "gpuResidentStrictParticles.dat"
            node_temperatures = tuple(500.0 + 100.0 * index for index in range(8))
            write_schema4_restart(restart, node_temperatures)
            state = MODULE.parse_particle_restart(restart)[0]
            properties = MODULE.ColdWallThermalProperties(
                2327.0, 20.0, 1.16e6, 1273.0
            )
            temperatures = MODULE.temperature_observables(state, properties)
            self.assertEqual(temperatures["cold_profile_active"], 1)
            self.assertAlmostEqual(
                temperatures["wall_contact_layer_temperature_k"], 500.0, places=4
            )
            self.assertAlmostEqual(
                temperatures["free_surface_temperature_k"], 1200.0, places=4
            )
            self.assertAlmostEqual(
                temperatures["axial_node_mean_temperature_k"], 850.0, places=4
            )

    def test_schema5_exposes_area_averaged_2d_boundary_temperatures(self):
        with tempfile.TemporaryDirectory() as directory:
            restart = Path(directory) / "gpuResidentStrictParticles.dat"
            node_temperatures = tuple(
                500.0 + 10.0 * radial + 100.0 * axial
                for radial in range(8)
                for axial in range(8)
            )
            write_schema5_restart(restart, node_temperatures)
            state = MODULE.parse_particle_restart(restart)[0]
            properties = MODULE.ColdWallThermalProperties(
                2327.0, 20.0, 1.16e6, 1273.0
            )
            temperatures = MODULE.temperature_observables(state, properties)
            self.assertEqual(temperatures["cold_profile_active"], 2)
            self.assertAlmostEqual(
                temperatures["wall_contact_layer_temperature_k"], 535.0, places=3
            )
            self.assertAlmostEqual(
                temperatures["free_surface_temperature_k"], 1235.0, places=3
            )
            self.assertAlmostEqual(
                temperatures["axial_node_mean_temperature_k"], 885.0, places=3
            )
            self.assertAlmostEqual(state.cold_2d_frozen_area_m2, 0.2, places=6)

    def test_legacy_restart_falls_back_to_solver_lumped_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            restart = Path(directory) / "gpuResidentStrictParticles.dat"
            write_restart(restart)
            state = MODULE.parse_particle_restart(restart)[0]
            properties = MODULE.ColdWallThermalProperties(
                2327.0, 20.0, 1.16e6, 1273.0
            )
            temperatures = MODULE.temperature_observables(state, properties)
            self.assertEqual(temperatures["cold_profile_active"], 0)
            self.assertEqual(
                temperatures["wall_contact_layer_temperature_k"],
                state.temperature_k,
            )


if __name__ == "__main__":
    unittest.main()
