#!/usr/bin/env python3
                                                                       

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINARY = (
    ROOT.parents[4]
    / "platforms/linux64GccDPInt32Opt/bin/CHT"
)


class CouplingLoggingContract(unittest.TestCase):
    def test_production_thermal_cadence_is_point_one_second(self) -> None:
        coupling = (
            ROOT.parents[1]
            / "examples/thermal/MSS7_laminar/constant/solidRegionProperties"
        ).read_text(encoding="utf-8")
        self.assertRegex(coupling, r"couplingInterval\s+0\.1\s*;")
        self.assertRegex(
            coupling, r"solidContactCouplingInterval\s+0\.1\s*;"
        )

    def test_every_completed_exchange_is_flushed(self) -> None:
        source = (ROOT / "diluteUgkwpFoam.C").read_text(encoding="utf-8")
        match = re.search(
            r"if\s*\(coupling\.coupled\)\s*\{(?P<body>.*?)\n\s*\}",
            source,
            re.S,
        )
        self.assertIsNotNone(match, "missing coupling.coupled progress block")
        body = match.group("body")
        for token in (
            'Info<< "runTime = "',
            '" simulationTime = "',
            '" particleCount = "',
            '" CoMax = "',
            '" CHTdeltaT = "',
            '" radiationCoupled = "',
            "<< endl;",
        ):
            self.assertIn(token, body)
        self.assertNotRegex(body, r"if\s*\(coupling\.radiationCoupled\)")

    def test_installed_binary_is_fresh_when_requested(self) -> None:
        source_mtime = max(
            path.stat().st_mtime
            for path in (
                ROOT / "diluteUgkwpFoam.C",
                ROOT / "gpu/GpuResidentStrict.cu",
                ROOT / "thermal/GpuSolidThermalCoupler.C",
            )
        )
        binary_mtime = BINARY.stat().st_mtime if BINARY.is_file() else -1.0
        print(
            "binaryFresh="
            f"{int(binary_mtime >= source_mtime)} "
            f"sourceMtime={source_mtime:.6f} binaryMtime={binary_mtime:.6f}"
        )
        if os.environ.get("MSS7_REQUIRE_FRESH_BINARY") == "1":
            self.assertGreaterEqual(
                binary_mtime,
                source_mtime,
                "installed CHT executable predates compiled source",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
