#!/usr/bin/env python3

import os
import re
import unittest
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "UGKP_RELEASE_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()
EXAMPLES = ROOT / "examples"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ExampleReleaseContract(unittest.TestCase):
    def require_directories(self, *directories: Path) -> None:
        missing = [path for path in directories if not path.is_dir()]
        if missing:
            relative = ", ".join(str(path.relative_to(ROOT)) for path in missing)
            self.skipTest(
                f"clean UGKP source package omits heavy release resources: {relative}"
            )

    def test_gas_and_les_cases_use_ugkp_pure_gas_interface(self):
        gas_root = EXAMPLES / "gks_flux_validation"
        les_root = EXAMPLES / "les_validation"
        self.require_directories(gas_root, les_root)
        cases = [
            *gas_root.glob("*/system/controlDict"),
            *(gas_root / "acousticWave").glob(
                "N*/system/controlDict"
            ),
            *les_root.glob("*/system/controlDict"),
        ]
        self.assertTrue(cases)
        for control in cases:
            self.assertRegex(
                read(control), r"\bapplication\s+ugkp\.8\s*;"
            )
            properties = control.parents[1] / "constant" / "ugkwpProperties"
            text = read(properties)
            self.assertRegex(text, r"\bgpuResidentStrict\s+true\s*;")
            self.assertRegex(text, r"\bgpuResidentPureGasOnly\s+true\s*;")

    def test_twophase_cases_use_ugkp_v4_l1_configuration(self):
        suite = EXAMPLES / "twophaseflux"
        self.require_directories(suite)
        for name in ("dustyBox", "dustyWave", "windSandShockTube"):
            case = suite / name
            control = read(case / "system" / "controlDict")
            props = read(case / "constant" / "ugkwpProperties")
            restart = (case / "0" / "gpuResidentStrictParticles.dat").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertRegex(
                control, r"\bapplication\s+ugkp\.8\s*;"
            )
            self.assertRegex(props, r"\bgpuResidentStrict\s+true\s*;")
            self.assertRegex(props, r"\bgpuResidentPureGasOnly\s+false\s*;")
            self.assertRegex(props, r"\bgpuCsrCellLocalPath\s+true\s*;")
            self.assertRegex(props, r"\bgpuCsrWarpAggregatedBinning\s+false\s*;")
            self.assertRegex(props, r"\bgpuCsrHeavyReduction\s+false\s*;")
            self.assertGreaterEqual(len(restart), 2)
            header = restart[0].split()
            self.assertEqual(header[0], "UGKP_PARTICLES_SCHEMA4")
            self.assertEqual(len(header), 2)
            particle_count = int(header[1])
            self.assertGreaterEqual(particle_count, 0)
            self.assertEqual(len(restart) - 1, particle_count)
            for record in restart[1:]:
                self.assertEqual(len(record.split()), 13)

    def test_twophase_validation_asset_and_one_click_scripts_exist(self):
        suite = EXAMPLES / "twophaseflux"
        asset = ROOT / "assets" / "twophaseflux_validation"
        self.require_directories(suite, asset)
        required = [
            asset / "source" / "GpuResidentStrict.cu",
            asset / "SOURCE_SHA256SUMS",
        ]
        for path in required:
            self.assertTrue(path.is_file(), path)
        for name in ("dustyWave", "windSandShockTube"):
            case = suite / name
            for script in ("Allrun", "Allclean", "Allwmake", "Allrestore"):
                self.assertTrue((case / script).is_file())

    def test_non_pressure_validation_cases_are_clean(self):
        roots = [
            EXAMPLES / "gks_flux_validation",
            EXAMPLES / "les_validation",
            EXAMPLES / "twophaseflux",
        ]
        self.require_directories(*roots)
        numeric = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$")
        unwanted = []
        for root in roots:
            for path in root.rglob("*"):
                if path.is_dir() and numeric.fullmatch(path.name) and path.name != "0":
                    unwanted.append(path)
                if path.is_file() and (
                    path.name.startswith("log.")
                    or path.name.startswith("runtime.")
                    or path.name.startswith("check_") and path.suffix == ".json"
                    or "comparison" in path.stem and path.suffix == ".png"
                ):
                    unwanted.append(path)
        self.assertEqual(unwanted, [])

    def test_pressuretest_runner_targets_existing_runtime_switch_suite(self):
        suite = EXAMPLES / "pressuretest" / "GPU_twophase"
        self.require_directories(suite)
        runner = read(suite / "Allruncase")
        self.assertIn("cp_cst_test", runner)
        self.assertIn("Allrun.sparse", runner)
        self.assertIn("Allrun.dense_e1", runner)
        for stale in ("hllc", "rusanov", "slau2_2"):
            self.assertNotIn(f"    {stale}\n", runner)


if __name__ == "__main__":
    unittest.main()
