"""Self-contained scalar checks; deliberately no host cooperative/team runs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


class AffineDirectionalScalar(unittest.TestCase):
    def check_probe(self, name, scalar_recovery=False):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / name
            command = ["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-I", str(ROOT)]
            if scalar_recovery:
                # Diagnostic-only copy: force the production scalar recovery
                # path on the same capture. No option is added to production.
                header = ROOT / "common/gasWall/WallCoherentSolver.H"
                source = header.read_text()
                original = "if(valid&&mirrorAdmissible&&(currentNutBranches!=baseNutBranches||!pchipFdCompatible(cfg,w,probe)))"
                self.assertEqual(source.count(original), 1)
                copied = Path(temporary) / "scalar-recovery-exercise.H"
                copied.write_text(source.replace(original, "if(false && " + original[3:]))
                command += ["-I", str(header.parent),
                            '-DCOHERENT_SOLVER_HEADER="' + str(copied) + '"',
                            "-DCOHERENT_SCALAR_RECOVERY_EXERCISE=1"]
            command += [str(HERE / (name + ".cpp")), "-o", str(executable)]
            build = subprocess.run(command,
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            print(run.stdout, end="")

    def test_free_wall_species_and_legacy_continuation(self):
        self.check_probe("affine_species_scalar_probe")

    def test_captured_723_fresh_cache_roots(self):
        self.check_probe("directional_fd_scalar_probe")

    def test_actual_scalar_recovery_on_captured_state(self):
        self.check_probe("directional_fd_scalar_probe", scalar_recovery=True)


if __name__ == "__main__":
    unittest.main()
