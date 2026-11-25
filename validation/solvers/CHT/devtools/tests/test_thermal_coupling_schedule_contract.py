#!/usr/bin/env python3
                                                                  

from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/CHT')
CPP_TEST = (VALIDATION_ROOT / 'solvers/CHT/devtools/tests/cpp/test_thermal_coupling_schedule.cpp')
COUPLER = ROOT / "thermal/GpuSolidThermalCoupler.C"


class ThermalCouplingScheduleContract(unittest.TestCase):
    def test_rounded_write_time_still_triggers_coupling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ugkpcht-schedule-") as temporary:
            executable = Path(temporary) / "test_thermal_coupling_schedule"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-I",
                    str(ROOT / "thermal"),
                    str(CPP_TEST),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            completed = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
            self.assertIn("PASS", completed.stdout)

    def test_host_and_device_share_the_output_relative_clock(self) -> None:
        host = COUPLER.read_text(encoding="utf-8")
        device = (SOURCE_ROOT / "common/gpu/ThermalEventGraph.cu").read_text(encoding="utf-8")
        for source in (host, device):
            self.assertIn("nextFullOrContactEvent", source)
            self.assertIn("outputRelativeRadiationDue", source)
        self.assertIn("stepToEvent", (SOURCE_ROOT / "common/gpu/AdvanceState.cuh").read_text())
        self.assertNotIn("elapsedSinceCoupling + SMALL", host)
        self.assertNotIn("elapsedSinceRadiation + SMALL", host)


if __name__ == "__main__":
    unittest.main(verbosity=2)
