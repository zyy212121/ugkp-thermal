#!/usr/bin/env python3
                                                                          

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ScheduledInletTemperatureContract(unittest.TestCase):
    def test_temperature_and_tables_are_uploaded_once(self) -> None:
        frontend = (ROOT / "diluteUgkwpFoam.C").read_text(encoding="utf-8")
        header = (ROOT / "gpu/GpuResidentStrict.H").read_text(encoding="utf-8")
        backend = (ROOT / "gpu/GpuResidentStrict.cu").read_text(encoding="utf-8")

        call = re.search(
            r"resident\.configureScheduledInlet\s*\((?P<body>.*?)\);",
            frontend,
            re.S,
        )
        self.assertIsNotNone(call)
        self.assertIn("gpuResidentInletTemperature", call.group("body"))
        self.assertNotIn("updateScheduledInlet", frontend)
        scheduled_decl = re.search(
            r"ugkwpGpuResidentStrictConfigureScheduledInlet\s*\((?P<body>[\s\S]*?)\);",
            header,
        )
        self.assertIsNotNone(scheduled_decl)
        self.assertIn("inletTemperature", scheduled_decl.group("body"))
        self.assertIn("pressureTimes", scheduled_decl.group("body"))
        self.assertIn("volumeFractionTimes", scheduled_decl.group("body"))
        self.assertIn("const double* gasBoundaryT", header)
        self.assertIn(
            "copyToDevice(s->gasBoundaryT, gasBoundaryT",
            backend,
        )
        self.assertIn(
            "copyToDevice(s->riemannBoundaryT, gasBoundaryT",
            backend,
        )

    @unittest.skip(
        "historical temperature-scan driver is intentionally excluded from "
        "the minimal runnable validation case"
    )
    def test_scan_edits_nested_patch_entries_not_literal_dotted_keys(self) -> None:
        script = (ROOT / "MSS7_pureGasCHT/run_temperature_scan.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("'boundaryField/inlet/value'", script)
        self.assertNotIn("-entry boundaryField.inlet.value", script)

        for relative in ("1/fluid/T", "1/fluid/Tp", "1/fluid/rho"):
            field = (ROOT / "MSS7_pureGasCHT" / relative).read_text(
                encoding="utf-8"
            )
            self.assertNotRegex(field, r"(?m)^boundaryField\.inlet\.value\b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
