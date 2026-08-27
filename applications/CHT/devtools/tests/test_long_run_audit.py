import tempfile
import unittest
from pathlib import Path

from devtools.audit_cht_long_run import (
    CONTINUOUS,
    TEMPERATURE_BINS,
    histogram,
    particle_values_are_finite,
    validate_distinct_run_inputs,
    validate_physical_parameters,
    validate_solver_log,
    within_envelope,
)


class LongRunAuditTests(unittest.TestCase):
    def test_baseline_repeat_envelope_is_fail_closed(self):
        self.assertTrue(within_envelope(100.0, 100.0, 100.0000001)[0])
        self.assertFalse(within_envelope(100.0, 100.0, 100.001)[0])

    def test_fixed_histogram_boundary_is_deterministic(self):
        self.assertEqual(histogram(999.0, TEMPERATURE_BINS), 5)
        self.assertEqual(histogram(1000.0, TEMPERATURE_BINS), 6)

    def test_solver_log_requires_one_finite_cht_exchange(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "solver.log"
            log.write_text("runTime = 1 CHTdeltaT = 1e-6\n", encoding="utf-8")
            validate_solver_log(log)
            log.write_text("runTime = 1 CHTdeltaT = 1e-6 value=inf\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_solver_log(log)

    def test_every_continuous_particle_column_must_be_finite(self):
        values = {name: 1.0 for name in CONTINUOUS}
        self.assertTrue(particle_values_are_finite(values))
        for name in CONTINUOUS:
            with self.subTest(name=name):
                candidate = dict(values)
                candidate[name] = float("nan")
                self.assertFalse(particle_values_are_finite(candidate))

    def test_long_run_physical_parameters_are_frozen(self):
        validate_physical_parameters(5e-9, 1500.0)
        for parcel_mass, particle_cp in ((1e-9, 1500.0), (5e-9, 1200.0)):
            with self.subTest(parcel_mass=parcel_mass, particle_cp=particle_cp):
                with self.assertRaises(ValueError):
                    validate_physical_parameters(parcel_mass, particle_cp)

    def test_same_case_cannot_fill_multiple_long_run_roles(self):
        inputs = {
            "baseline": (Path("case-a"), Path("log-a")),
            "baseline_repeat": (Path("case-a"), Path("log-b")),
            "candidate": (Path("case-c"), Path("log-c")),
        }
        with self.assertRaisesRegex(ValueError, "case paths must be distinct"):
            validate_distinct_run_inputs(inputs)


if __name__ == "__main__":
    unittest.main()
