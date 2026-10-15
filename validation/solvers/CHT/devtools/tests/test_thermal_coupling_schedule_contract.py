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

    def test_contact_uses_tolerant_time_schedule_and_radiation_uses_sequence(self) -> None:
        source = COUPLER.read_text(encoding="utf-8")
        self.assertIn('#include "ThermalCouplingSchedule.H"', source)
        self.assertEqual(source.count("ugkpcht::thermalEventIsDue"), 1)
        self.assertIn("radiationCouplingFrequency", source)
        self.assertRegex(
            source,
            r"decision\.exchangeSequence[\s\S]{0,160}?%[\s\S]{0,160}?radiationCouplingFrequency",
        )
        self.assertNotIn("radiationCouplingInterval", source)
        self.assertNotIn("elapsedSinceCoupling + SMALL", source)
        self.assertNotIn("elapsedSinceRadiation + SMALL", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
