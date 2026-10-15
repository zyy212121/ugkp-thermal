#!/usr/bin/env python3
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

                                                                      

from source_contract_utils import read_source
from pathlib import Path
import unittest


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')
ACTIVE = (
    ROOT / "gpu" / "GpuResidentStrict.H",
    ROOT / "gpu" / "GpuBackendApi.H",
    ROOT / "gpu" / "GpuBackendClient.C",
    ROOT / "gpu" / "GpuBackendProtocol.H",
    ROOT / "private_backend" / "GpuBackendServer.C",
    ROOT / "private_backend" / "GpuResidentStrict.cu",
)


class NoWallRetentionContract(unittest.TestCase):
    def test_removed_state_and_configuration_are_absent(self) -> None:
        active = "\n".join(path.read_text(encoding="utf-8") for path in ACTIVE)
        self.assertIn("#define GPU_OPERATOR_THERMAL 0", (ROOT / "private_backend/GpuResidentStrict.cu").read_text(encoding="utf-8"))
        forbidden = (
            "pStuck",
            "particleStuck",
            "ParticleStuck",
            "Sommerfeld",
            "sommerfeld",
            "gpuResidentStuck",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, active)

    def test_restart_output_contains_mobile_particle_fields(self) -> None:
        frontend = read_source(ROOT / "gpu" / "GpuResidentStrict.H")

        self.assertIn("UGKP_PARTICLES_SCHEMA1_BIN", frontend)
        for field in ("pCellId", "pStatus", "pRng", "pOrigId"):
            self.assertIn(field, frontend)

    def test_existing_reflection_coefficients_remain(self) -> None:
        frontend = read_source(ROOT / "gpu" / "GpuResidentStrict.H")
        cuda = read_source(ROOT / "private_backend" / "GpuResidentStrict.cu")
        self.assertIn("particleWallCoeffs", frontend)
        self.assertIn("cellFaceRestitution", frontend)
        self.assertIn("const double restitution = s.cellFaceRestitution[plane]", cuda)


if __name__ == "__main__":
    unittest.main()
