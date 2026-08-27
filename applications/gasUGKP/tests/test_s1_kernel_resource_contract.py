#!/usr/bin/env python3
                                                                      

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = (
    ROOT.parents[4]
    / "platforms"
    / "linux64GccDPInt32Opt"
    / "bin"
    / "gasUGKPCudaBackend"
)
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")


def kernel_resources(kernel_name: str) -> list[dict[str, int | str]]:
    if not BACKEND.is_file():
        raise AssertionError(f"UGKP backend is missing: {BACKEND}")
    if not CUOBJDUMP.is_file():
        raise AssertionError(f"cuobjdump is missing: {CUOBJDUMP}")
    result = subprocess.run(
        [str(CUOBJDUMP), "--dump-resource-usage", str(BACKEND)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pattern = re.compile(
        rf"Function\s+(?P<symbol>[^\n]*{re.escape(kernel_name)}[^\n]*):\n"
        r"\s+REG:(?P<reg>\d+)\s+STACK:(?P<stack>\d+)\s+"
        r"SHARED:(?P<shared>\d+)\s+LOCAL:(?P<local>\d+)"
    )
    return [
        {
            "symbol": match.group("symbol"),
            "reg": int(match.group("reg")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in pattern.finditer(result.stdout)
    ]


class S1KernelResourceContract(unittest.TestCase):
    def test_l0_full_collision_pool_matches_the_established_resource_budget(self) -> None:
        resources = kernel_resources("accumulatePoissonPoolParticlesByCellKernel")
        self.assertEqual(len(resources), 1, resources)
        symbols = tuple(str(resource["symbol"]) for resource in resources)
        self.assertTrue(any("ILb0EE" in symbol for symbol in symbols), symbols)
        self.assertFalse(any("ILb1EE" in symbol for symbol in symbols), symbols)
        for resource in resources:
            self.assertEqual(resource["stack"], 0, resource)
            self.assertEqual(resource["local"], 0, resource)
            self.assertLessEqual(resource["reg"], 48, resource)

    def test_split_collision_segments_are_spill_free_and_s1_is_lean(self) -> None:
        resources = kernel_resources(
            "accumulatePoissonPoolSplitSegmentByCellKernel"
        )
        self.assertEqual(len(resources), 4, resources)
        s1_symbols = ("ILb1ELb0ELb0EE", "ILb0ELb1ELb0EE")
        for resource in resources:
            self.assertEqual(resource["stack"], 0, resource)
            self.assertEqual(resource["local"], 0, resource)
            if any(token in resource["symbol"] for token in s1_symbols):
                self.assertLessEqual(resource["reg"], 48, resource)

    def test_pressure_projection_has_two_spill_free_specialisations(self) -> None:
        resources = kernel_resources("applyCollisionalPressureProjectionKernel")
        self.assertEqual(len(resources), 2, resources)
        for resource in resources:
            self.assertEqual(resource["stack"], 0, resource)
            self.assertEqual(resource["local"], 0, resource)
            self.assertLessEqual(resource["reg"], 64, resource)


if __name__ == "__main__":
    unittest.main(verbosity=2)
