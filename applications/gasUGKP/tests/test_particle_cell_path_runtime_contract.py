#!/usr/bin/env python3
                                                                     

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="strict")


class ParticleCellPathRuntimeContract(unittest.TestCase):
    def test_switch_crosses_every_process_boundary(self) -> None:
        wrapper = source("gpu/GpuResidentStrict.H")
        api = source("gpu/GpuBackendApi.H")
        protocol = source("gpu/GpuBackendProtocol.H")
        client = source("gpu/GpuBackendClient.C")
        server = source("private_backend/GpuBackendServer.C")
        cuda = source("private_backend/GpuResidentStrict.cu")

        shared = source("../../common/GpuSchedulingConfiguration.H")
        self.assertIn('"gpuCsrLevel"', shared)
        for text in (api, protocol, client, server, cuda):
            self.assertIn("csrCellLocalPathEnabled", text)
        self.assertIn("magic = 0x47554750U;", protocol)
        self.assertIn("protocolMajor = 7", protocol)
        self.assertIn("protocolMinor = 0", protocol)

    def test_public_level_derives_a_valid_internal_bundle(self) -> None:
        shared = source("../../common/GpuSchedulingConfiguration.H")
        client = source("gpu/GpuBackendClient.C")
        server = source("private_backend/GpuBackendServer.C")
        cuda = source("private_backend/GpuResidentStrict.cu")

        self.assertIn("GpuCsrLevel::L2", shared)
        self.assertIn("GpuCsrLevel::automatic", shared)
        for text in (client, server, cuda):
            self.assertIn("csrCellLocalPathEnabled", text)
            self.assertIn("csrHeavyReductionMode", text)
            self.assertIn("csrWarpAggregatedBinning", text)
        self.assertIn(
            "value.csrCellLocalPath = value.csrLevel != GpuCsrLevel::L0",
            shared,
        )
        self.assertIn(
            "value.warpAggregatedBinning = value.csrLevel != GpuCsrLevel::L0",
            shared,
        )
        for text in (client, server, cuda):
            self.assertRegex(
                text,
                r"!\s*(?:a\.)?csrCellLocalPathEnabled[\s\S]{0,160}"
                r"csrHeavyReductionMode[\s\S]{0,160}"
                r"csrWarpAggregatedBinning",
            )

    def test_both_execution_paths_are_present(self) -> None:
        cuda = source("private_backend/GpuResidentStrict.cu")
        for symbol in (
            "applyCollisionalPressureProjectionCellAtomicKernel",
            "applyCollisionalPressureProjectionParticlesAtomicKernel",
            "accumulateParticlePoolAtomicKernel",
            "clearParticleMomentsAndCountsAtomicKernel",
            "accumulateParticleMomentsAtomicKernel",
            "normalizeParticleMomentsAtomicKernel",
            "gatherSelectedParticlesKernel",
            "commitSelectedParticleBuffersKernel",
        ):
            self.assertIn(symbol, cuda)

        self.assertIn("if (s->csrCellLocalPathEnabled != 0)", cuda)
        self.assertRegex(
            cuda,
            r"if \(s->csrCellLocalPathEnabled != 0\)[\s\S]{0,2500}else",
        )
        self.assertIn("binParticlesByCell", cuda)
        self.assertIn("accumulateParticleMomentsSegmentedKernel", cuda)
        self.assertIn("gatherCellLocalParticlesKernel", cuda)

    def test_runtime_banner_identifies_the_selected_path(self) -> None:
        wrapper = source("gpu/GpuResidentStrict.H")
        self.assertIn("Particle cell path: csrCellLocalPath=", wrapper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
