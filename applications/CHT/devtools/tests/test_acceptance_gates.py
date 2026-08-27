import tempfile
import unittest
from pathlib import Path

from devtools.compare_cht_particle_restart import gate_failures as particle_failures
from devtools.compare_cht_time_directories import gate_failures as field_failures
from devtools.summarize_cht_performance import (
    KernelRun,
    evaluate_gates,
    tree_sha256,
    validate_raw_evidence_identity,
    validate_solver_log,
)


def particle_result(*, continuous_exact: bool = True, max_scaled: float = 0.0):
    return {
        "id_sets_equal": True,
        "missing_ids": 0,
        "extra_ids": 0,
        "discrete_exact": True,
        "discrete_mismatches": {"cell": 0, "status": 0, "rng": 0},
        "cell_occupancy_l1": 0,
        "continuous_all_exact": continuous_exact,
        "continuous": {"x": {"max_scaled": max_scaled}},
    }


class ParticleAcceptanceTests(unittest.TestCase):
    def test_strict_gate_rejects_continuous_difference(self):
        failures = particle_failures(
            particle_result(continuous_exact=False, max_scaled=1.0e-14), "strict"
        )
        self.assertTrue(any("continuous" in failure for failure in failures))

    def test_tolerance_gate_passes_and_rejects_at_boundary(self):
        result = particle_result(continuous_exact=False, max_scaled=1.0e-12)
        self.assertEqual(particle_failures(result, "tolerance", 2.0e-9), [])
        self.assertTrue(particle_failures(result, "tolerance", 1.0e-13))

    def test_discrete_gate_rejects_missing_id(self):
        result = particle_result()
        result["id_sets_equal"] = False
        result["missing_ids"] = 1
        self.assertTrue(particle_failures(result, "discrete"))


class FieldAcceptanceTests(unittest.TestCase):
    def test_strict_gate_rejects_numeric_difference(self):
        result = {"file_sets_equal": True, "all_numeric_exact": False, "max_scaled": 1e-16}
        self.assertTrue(field_failures(result, "strict"))

    def test_tolerance_gate_is_bounded(self):
        result = {"file_sets_equal": True, "all_numeric_exact": False, "max_scaled": 1e-12}
        self.assertEqual(field_failures(result, "tolerance", 2e-9), [])
        self.assertTrue(field_failures(result, "tolerance", 1e-13))


def kernel_runs(v1_pool: float = 90.0) -> list[KernelRun]:
    runs = []
    for round_id, offset in ((1, -1.0), (2, 0.0), (3, 1.0)):
        runs.append(KernelRun(round_id, "baseline", 100.0 + offset, 50.0, 30.0, 1000.0))
        runs.append(KernelRun(round_id, "v1", v1_pool + offset, 50.0, 30.0, 900.0))
        runs.append(KernelRun(round_id, "v2", 89.0 + offset, 40.0, 30.0, 990.0))
    return runs


class PerformanceAcceptanceTests(unittest.TestCase):
    def test_all_frozen_gates_pass(self):
        kernel_summary = {
            "baseline": {"pool_median_ns": 100.0, "total_cuda_kernel_ns": 1000.0},
            "v1": {"pool_median_ns": 90.0, "relax_median_ns": 50.0},
            "v2": {"relax_median_ns": 40.0, "total_cuda_kernel_ns": 990.0},
        }
        wall_summary = {"baseline": {"median": 10.0}, "v2": {"median": 10.1}}
        metrics, failures = evaluate_gates(kernel_runs(), kernel_summary, wall_summary)
        self.assertEqual(failures, [])
        self.assertTrue(metrics["v1_pool_ranges_separated"])

    def test_regressions_cannot_be_reported_as_pass(self):
        kernel_summary = {
            "baseline": {"pool_median_ns": 100.0, "total_cuda_kernel_ns": 1000.0},
            "v1": {"pool_median_ns": 98.0, "relax_median_ns": 50.0},
            "v2": {"relax_median_ns": 51.0, "total_cuda_kernel_ns": 1030.0},
        }
        wall_summary = {"baseline": {"median": 10.0}, "v2": {"median": 10.3}}
        _, failures = evaluate_gates(
            kernel_runs(v1_pool=98.0), kernel_summary, wall_summary
        )
        self.assertGreaterEqual(len(failures), 5)

    def test_nonfinite_solver_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "solver.log"
            log.write_text(
                "runTime = 1 simulationTime = 1 CHTdeltaT = 1e-6\nvalue = nan\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_solver_log(log)

    def test_truncated_wall_evidence_cannot_be_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "wall-r1-baseline"
            log_dir.mkdir()
            (log_dir / "solver.log").write_text(
                "runTime = 1 CHTdeltaT = 1e-6\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                validate_raw_evidence_identity(log_dir)

    def test_unlisted_case_input_changes_tree_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            for name in ("system", "constant", "0.10000002"):
                (case / name).mkdir()
            extra = case / "constant" / "unlistedPhysicsInput"
            extra.write_text("value 1;\n", encoding="utf-8")
            before = tree_sha256(case, ("system", "constant", "0.10000002"))
            extra.write_text("value 2;\n", encoding="utf-8")
            after = tree_sha256(case, ("system", "constant", "0.10000002"))
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
