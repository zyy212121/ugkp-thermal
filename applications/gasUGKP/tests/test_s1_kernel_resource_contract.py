#!/usr/bin/env python3
                                                                      

from __future__ import annotations

from pathlib import Path
import re
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GAS_UGKP_RESOURCE_BACKEND", str(ROOT.parents[4] / "platforms/linux64GccDPInt32Opt/bin/gasUGKPCudaBackend")))
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

    def test_active_pressure_cache_kernels_are_spill_free(self) -> None:
        # Validate this package's active kernels, not an unrelated engineering
        # installation or the old, no-longer-launched template specialisations.
        for name, register_limit in (
            ("preparePressureProjectionCacheKernel", 80),
            ("applyCachedPressureProjectionParticlesKernel", 48),
        ):
            resources = kernel_resources(name)
            self.assertEqual(len(resources), 1, resources)
            for resource in resources:
                self.assertEqual(resource["stack"], 0, resource)
                self.assertEqual(resource["local"], 0, resource)
                self.assertLessEqual(resource["reg"], register_limit, resource)

    def test_uncached_drag_and_pool_clear_are_spill_free(self) -> None:
        clear = kernel_resources("clearPoissonThermalPoolKernel")
        relax = kernel_resources("relaxParticlesToResidentGasKernel")
        self.assertEqual(len(clear), 1, clear)
        self.assertEqual(len(relax), 2, relax)
        for resource in clear + relax:
            self.assertEqual(resource["stack"], 0, resource)
            self.assertEqual(resource["local"], 0, resource)
            if resource in clear:
                limit = 42
            else:
                limit = 58
            self.assertLessEqual(resource["reg"], limit, resource)


if __name__ == "__main__":
    unittest.main(verbosity=2)
