#!/usr/bin/env python3
                                                                

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "gpu" / "GpuResidentStrict.cu"
CPP_TEST = ROOT / "devtools" / "tests" / "cpp" / "test_openfoam_wall_functions.cpp"


class WaleWallFunctionContract(unittest.TestCase):
    def test_openfoam_spalding_algebra(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ugkpcht-wall-") as temporary:
            executable = Path(temporary) / "test_openfoam_wall_functions"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-I",
                    str(ROOT / "gpu"),
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

    def test_flux_and_diffusion_share_physical_wall_transport(self) -> None:
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        self.assertIn('#include "OpenFoamWallFunctions.cuh"', source)
        self.assertRegex(
            source,
            r"gasFaceSubgridTransportProperties\s*\(\s*"
            r"s,\s*f,\s*own,\s*nei,\s*boundaryKind,",
        )
        self.assertGreaterEqual(
            len(re.findall(r"gasFaceSubgridTransportProperties\s*\(", source)),
            3,
            "definition, face-flux call, and diffusion call must all exist",
        )
        self.assertRegex(
            source,
            r"boundaryKind\s*==\s*2[\s\S]{0,1800}"
            r"ugkpwall::spaldingWallState",
        )
        self.assertIn("wallSubgridTransport", source)


if __name__ == "__main__":
    unittest.main()
