#!/usr/bin/env python3
                                                                

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA_PATH = ROOT / "private_backend/GpuResidentStrict.cu"


def cuda_source() -> str:
    return CUDA_PATH.read_text(encoding="utf-8", errors="strict")


def function_body(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


class UGKPSparseStuckFinalizationContract(unittest.TestCase):
    def test_sampling_captures_only_selected_stuck_indices_without_rng(self) -> None:
        cuda = cuda_source()
        body = function_body(
            cuda,
            "__device__ void appendSelectedStuckParticleIndex",
            "__device__ void sampleOnePoissonPoolParticle",
        )
        self.assertIn("if (s.pStuck[i] == 0)", body)
        self.assertIn("atomicAdd(s.compactCountDevice, 1)", body)
        self.assertIn("s.compactPStatus[slot] = i;", body)
        self.assertNotIn("uniform01Device", body)
        self.assertNotIn("normalPairDevice", body)

    def test_mobile_kernel_excludes_capillary_slow_path(self) -> None:
        cuda = cuda_source()
        body = function_body(
            cuda,
            "__global__ void correctPoissonThermalizedMobileParticlesKernel",
            "__global__ void correctPoissonThermalizedStuckParticlesKernel",
        )
        self.assertIn("s.pStuck[i] != 0", body)
        self.assertIn("correctOnePoissonThermalizedMobileParticle", body)
        self.assertNotIn("evaluateCapillaryDetachmentState", body)
        self.assertNotIn("applyCapillaryContactDamage", body)
        self.assertNotIn("pDepositionArea", body)

    def test_stuck_kernel_consumes_sparse_index_scratch(self) -> None:
        cuda = cuda_source()
        body = function_body(
            cuda,
            "__global__ void correctPoissonThermalizedStuckParticlesKernel",
            "__global__ void gatherCellLocalParticlesKernel",
        )
        self.assertIn("*s.compactCountDevice", body)
        self.assertIn("const int i = s.compactPStatus[pos];", body)
        self.assertIn("correctOnePoissonThermalizedStuckParticle", body)
        self.assertNotIn("uniform01Device", body)
        self.assertNotIn("normalPairDevice", body)

    def test_single_pool_pass_resets_and_launches_ordered_paths(self) -> None:
        cuda = cuda_source()
        advance = function_body(
            cuda,
            'extern "C" int ugkwpGpuResidentStrictAdvance',
            'extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly',
        )
        self.assertEqual(
            advance.count(
                "cudaMemset(s->compactCountDevice, 0, sizeof(int))"
            ),
            1,
        )
        self.assertEqual(
            advance.count(
                "correctPoissonThermalizedMobileParticlesKernel<<<"
            ),
            1,
        )
        self.assertEqual(
            advance.count(
                "correctPoissonThermalizedStuckParticlesKernel<<<"
            ),
            1,
        )
        remaining = advance
        for _ in range(1):
            mobile = remaining.index(
                "correctPoissonThermalizedMobileParticlesKernel<<<"
            )
            stuck = remaining.index(
                "correctPoissonThermalizedStuckParticlesKernel<<<",
                mobile,
            )
            self.assertLess(mobile, stuck)
            remaining = remaining[stuck + 1 :]

    def test_sparse_path_reuses_existing_compaction_storage(self) -> None:
        cuda = cuda_source()
        state = function_body(
            cuda,
            "struct DeviceState",
            "struct ActiveParticleIndexPredicate",
        )
        self.assertNotIn("selectedStuckIndex", state)
        self.assertNotIn("selectedStuckCount", state)
        self.assertIn("int* compactPStatus", state)
        self.assertIn("int* compactCountDevice", state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
