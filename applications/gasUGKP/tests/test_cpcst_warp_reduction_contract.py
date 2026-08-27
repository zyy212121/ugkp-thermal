#!/usr/bin/env python3
                                                                       

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "private_backend" / "GpuResidentStrict.cu"


class CpcstWarpReductionContract(unittest.TestCase):
    def test_component_reduction_uses_a_converged_full_warp(self) -> None:
        source = CUDA.read_text(encoding="utf-8")
        start = source.index("template<int NumComponents>")
        end = source.index(
            "__global__ void accumulatePoissonPoolParticlesByCellKernel",
            start,
        )
        body = source[start:end]

        self.assertIn(
            "constexpr unsigned int fullWarpMask = 0xffffffffu;",
            body,
        )
        self.assertIn("__syncwarp(fullWarpMask);", body)
        self.assertIn("if ((blockDim.x & 31) != 0)", body)
        self.assertNotIn("= __activemask();", body)
        self.assertNotIn("__popc(", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
