from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class UGKPIdentityContract(unittest.TestCase):
    def test_frontend_backend_and_protocol_are_isolated(self) -> None:
        make = read("Make/files")
        client = read("gpu/GpuBackendClient.C")
        protocol = read("gpu/GpuBackendProtocol.H")
        build = read("private_backend/build_private_backend.sh")
        banner = read("diluteUgkwpFoam.C")

        self.assertIn("$(FOAM_USER_APPBIN)/gasUGKP", make)
        self.assertIn("GAS_UGKP_CUDA_BACKEND", client)
        self.assertIn("gasUGKPCudaBackend", client)
        self.assertIn("0x47554750U", protocol)
        self.assertIn("protocolMajor = 7", protocol)
        self.assertIn("GAS_UGKP_FMAD", build)
        self.assertIn("gasUGKP:", banner)
        joined = "\n".join((make, client, protocol, build, banner))
        for stale in ("legacySolverName", "legacyCudaBackend"):
            self.assertNotIn(stale, joined)


class UGKPBlockConfigurationContract(unittest.TestCase):
    def test_b2_b3_are_defaulted_validated_and_forwarded(self) -> None:
        wrapper = read("gpu/GpuResidentStrict.H")
        shared = read("../../common/GpuSchedulingConfiguration.H")
        api = read("gpu/GpuBackendApi.H")
        protocol = read("gpu/GpuBackendProtocol.H")
        client = read("gpu/GpuBackendClient.C")
        server = read("private_backend/GpuBackendServer.C")
        cuda = read("private_backend/GpuResidentStrict.cu")

        self.assertIn("gpuParticleBlockThreads", shared)
        self.assertIn("gpuReductionBlockThreads", shared)
        for text in (api, protocol, client, server, cuda):
            self.assertIn("particleBlockThreads", text)
            self.assertIn("reductionBlockThreads", text)
        self.assertIn("sizeof(CreateArgs) == 392", protocol)
        self.assertIn("s->particleBlockThreads = particleBlockThreads", cuda)
        self.assertIn("s->reductionBlockThreads = reductionBlockThreads", cuda)

    def test_global_launches_do_not_retain_an_unclassified_256_block(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertNotIn("const int block = 256;", cuda)
        self.assertNotIn("const int preparationBlock = 256;", cuda)
        for value in (32, 64, 128, 256):
            self.assertIn(f"gatherCellLocalParticlesKernel<{value}>", cuda)
        self.assertIn("cub::BlockScan<int, BlockThreads>", cuda)

    def test_cuda_occupancy_is_the_launch_limit_source(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", cuda)
        self.assertIn("cudaDevAttrMaxThreadsPerBlock", cuda)
        self.assertIn("cudaDevAttrMaxBlocksPerMultiprocessor", cuda)
        self.assertIn("s->multiprocessorCount*s->particleBlocksPerSm", cuda)
        self.assertIn("s->multiprocessorCount*s->heavyBlocksPerSm", cuda)


class UGKPHeavyContract(unittest.TestCase):
    def test_s1_has_static_heavy_disabled_light_kernels(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn("accumulatePoissonPoolParticlesByCellKernel<false>", cuda)
        self.assertIn("accumulateParticleMomentsSegmentedKernel<false>", cuda)
        self.assertIn(
            "accumulatePoissonPoolSplitSegmentByCellKernel<true, false, false>",
            cuda,
        )
        allocation = re.search(
            r"if \(s->csrHeavyReductionEnabled != 0 && s->particleCapacity > 0\)"
            r"(?P<body>[\s\S]*?)rc \|= allocate\(s->compactPx",
            cuda,
        )
        self.assertIsNotNone(allocation)
        self.assertIn("csrHeavyTaskCount", allocation.group("body"))

    def test_dynamic_threshold_and_exact_split_population(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn("configureDynamicCsrHeavyPolicyKernel", cuda)
        self.assertIn("s.reductionBlockThreads", cuda)
        self.assertIn("s.multiprocessorCount", cuda)
        self.assertIn("s.lightBlocksPerSm", cuda)
        self.assertIn("s.preBaseCellOffset[s.nCells]", cuda)
        self.assertIn("s.cellParticleOffset[s.nCells]", cuda)
        self.assertIn("baseCount + injectionCount <= s.csrHeavyCellThreshold", cuda)
        self.assertIn("s.csrHeavyTileParticles = static_cast<int>(threshold)", cuda)
        self.assertIn("s->nBoundarySources == 0", cuda)

    def test_probe_schema_records_dynamic_geometry(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn('std::fprintf(file, "3,")', cuda)
        for column in (
            "block_exponent",
            "block_threads",
            "sm_count",
            "particle_blocks_per_sm",
            "light_blocks_per_sm",
            "heavy_blocks_per_sm",
            "heavy_reduction_enabled",
        ):
            self.assertIn(column, cuda)


if __name__ == "__main__":
    unittest.main()
