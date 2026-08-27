#!/usr/bin/env python3
                                                                      

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COUPLER = ROOT / "thermal" / "GpuSolidThermalCoupler.C"


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


if __name__ == "__main__":
    unittest.main()
