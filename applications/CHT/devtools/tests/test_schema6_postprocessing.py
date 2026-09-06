import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples/thermal/postprocessing"))
sys.path.insert(0, str(ROOT / "applications/CHT/devtools"))
from fsh_restart_reader import iter_fsh_chunks
from bentsrm_case_reader import particle_chunks
from compare_cht_particle_restart import compare, gate_failures

spec = importlib.util.spec_from_file_location("single_drop_schema6", ROOT / "examples/thermal/singleAluminaDrop/assets/postprocessing/single_alumina_drop_validation.py")
drop = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = drop
spec.loader.exec_module(drop)

TIME = 1.0 + 2.0 ** -40


def fixture(path, schema, node_delta=0.0):
    time_code = "d" if schema == 6 else "f"
    with path.open("wb") as out:
        out.write(f"UGKP_FSH_PARTICLES_SCHEMA{schema}_BIN 3 2\n".encode())
        for first, count in ((0, 2), (2, 1)):
            out.write(struct.pack("<I", count))
            def field(code, values):
                out.write(struct.pack("<" + str(len(values)) + code, *values))
            for value in (0.1, 0.2, 0.3, 2.0, 3.0, 4.0, 2373.15, 0.25, 1e-4, 1e-10):
                field("d", [value] * count)
            for code, value in (("i", 0), ("i", 1), ("Q", 123)):
                field(code, [value] * count)
            field("Q", [2 ** 54 + i for i in range(first, first + count)])
            for code, value in (("B", 3), ("i", 7), ("f", 0.0)):
                field(code, [value] * count)
            if schema >= 2:
                field(time_code, [TIME] * count)
                field("f", [1e-8] * count)
            if schema >= 3:
                field("f", [0.25] * count)
            if schema >= 4:
                field("f", [1e6 + i for _ in range(count) for i in range(8)])
                field("f", [0.0] * (count * 8))
                field("f", [0.0] * count)
                field(time_code, [TIME] * count)
            if schema >= 5:
                field("f", [2e6 + i + node_delta for _ in range(count) for i in range(64)])
                field(time_code, [TIME] * (count * 8))
                field("f", [1e-9] * count)


class SchemaCompatibilityTests(unittest.TestCase):
    def test_multichunk_schema1_to6_and_exact_integer_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "particles.dat"
            for schema in range(1, 7):
                with self.subTest(schema=schema):
                    fixture(path, schema)
                    chunks = list(iter_fsh_chunks(path))
                    self.assertEqual([len(c["T"]) for c in chunks], [2, 1])
                    self.assertEqual(int(chunks[-1]["original_id"][0]), 2 ** 54 + 2)
                    plot_chunks = list(particle_chunks(path))
                    np.testing.assert_array_equal(chunks[0]["T"], plot_chunks[0]["T"])
                    if schema >= 2:
                        states = drop.parse_particle_restart(path)
                        self.assertEqual(len(states), 3)
                        self.assertEqual(states[-1].original_id, 2 ** 54 + 2)
                        expected_time = TIME if schema == 6 else 1.0
                        self.assertEqual(states[0].contact_duration_s, expected_time)
                    if schema == 6:
                        self.assertEqual(states[0].cold_contact_age_s, TIME)
                        self.assertEqual(states[0].cold_2d_ring_contact_ages_s, (TIME,) * 8)
                        self.assertEqual(states[-1].cold_2d_node_specific_enthalpies_j_kg[-1], 2000063.0)
                        self.assertGreater(drop.effective_contact_area(states[0]), 0.0)

    def test_reject_truncated_and_trailing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "particles.dat"
            fixture(path, 6)
            valid = path.read_bytes()
            for data in (valid[:-1], valid + b"x", valid[:80]):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    list(iter_fsh_chunks(path))
                with self.assertRaises(ValueError):
                    drop.parse_particle_restart(path)

    def test_comparison_detects_cold_profile_and_time_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.dat", Path(directory) / "b.dat"
            fixture(a, 6)
            fixture(b, 6)
            self.assertEqual(gate_failures(compare(a, b), "strict"), [])
            fixture(b, 6, node_delta=1.0)
            result = compare(a, b)
            self.assertEqual(result["continuous"]["cold2d_node_specific_enthalpy_63"]["max_abs"], 1.0)
            self.assertTrue(gate_failures(result, "strict"))
            fixture(b, 5)
            result = compare(a, b)
            self.assertEqual(result["continuous"]["contact_duration"]["max_abs"], 2.0 ** -40)


if __name__ == "__main__":
    unittest.main()
