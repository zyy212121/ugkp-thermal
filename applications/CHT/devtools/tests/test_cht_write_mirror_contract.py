#!/usr/bin/env python3
                                                                      

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COUPLER = ROOT / "thermal" / "GpuSolidThermalCoupler.C"
MAIN = ROOT / "diluteUgkwpFoam.C"
GPU_HEADER = ROOT / "gpu" / "GpuResidentStrict.H"


class ChtWriteMirrorContract(unittest.TestCase):
    def test_exchange_refreshes_host_fields_on_write_without_radiation(self) -> None:
        source = COUPLER.read_text(encoding="utf-8")
        pattern = re.compile(
            r"if\s*\(\s*radiationDue\s*\|\|\s*runTime\.writeTime\(\)\s*\)"
            r"[\s\S]{0,500}?resident\.downloadToHostMirror",
        )
        self.assertRegex(
            source,
            pattern,
            "a non-radiation CHT exchange at writeTime must refresh the gas mirror",
        )

    def test_persistence_downloads_if_no_exchange_completed_at_write(self) -> None:
        source = COUPLER.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"completedSimulationTimeS\s*!=\s*runTime\.value\(\)"
            r"[\s\S]{0,500}?resident\.downloadToHostMirror",
        )

    def test_negative_one_write_interval_is_owned_by_thermal_coupling(self) -> None:
        main = MAIN.read_text(encoding="utf-8")
        coupler = COUPLER.read_text(encoding="utf-8")
        gpu_header = GPU_HEADER.read_text(encoding="utf-8")
        self.assertIn("configuredWriteInterval == scalar(-1)", main)
        self.assertIn("runTime.setWriteInterval(GREAT)", main)
        self.assertRegex(
            main,
            r"runTime\.writeTime\(\)[\s\S]{0,160}?writeAtThermalCoupling"
            r"[\s\S]{0,80}?thermalCoupledThisStep",
        )
        self.assertRegex(
            coupler,
            r"!runTime\.writeTime\(\)[\s\S]{0,180}?completedSimulationTimeS"
            r"[\s\S]{0,80}?!=\s*runTime\.value\(\)",
        )
        self.assertIn("const bool allowThermalCouplingWrite = false", gpu_header)
        self.assertGreaterEqual(
            gpu_header.count("!runTime.writeTime() && !allowThermalCouplingWrite"),
            5,
        )
        self.assertIn(
            "stageThermalRestartMirrors\n        (\n            runTime,\n            thermalCouplingWrite",
            coupler,
        )
        self.assertRegex(
            coupler,
            r"thermalCouplingWrite[\s\S]{0,1800}?downloadToHostMirror"
            r"[\s\S]{0,400}?dMeanCell, true",
        )
        self.assertRegex(
            coupler,
            r"thermalCouplingWrite\s*&&\s*!runTime\.writeTime\(\)"
            r"[\s\S]{0,100}?runTime\.writeNow\(\)"
            r"[\s\S]{0,80}?runTime\.write\(\)",
        )


if __name__ == "__main__":
    unittest.main()
