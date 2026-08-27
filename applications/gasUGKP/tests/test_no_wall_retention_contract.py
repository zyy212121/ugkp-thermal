#!/usr/bin/env python3
                                                                      

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
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
        active = "\n".join(path.read_text() for path in ACTIVE)
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

    def test_restart_schema_contains_only_mobile_particle_state(self) -> None:
        frontend = (ROOT / "gpu" / "GpuResidentStrict.H").read_text()
        self.assertIn("UGKP_PARTICLES_SCHEMA4", frontend)
        self.assertNotIn("UGKP_PARTICLES_SCHEMA2", frontend)
        self.assertNotIn("UGKP_PARTICLES_SCHEMA3", frontend)

    def test_existing_reflection_coefficients_remain(self) -> None:
        frontend = (ROOT / "gpu" / "GpuResidentStrict.H").read_text()
        cuda = (ROOT / "private_backend" / "GpuResidentStrict.cu").read_text()
        self.assertIn("particleWallCoeffs", frontend)
        self.assertIn("cellFaceRestitution", frontend)
        self.assertIn("const double restitution = s.cellFaceRestitution[plane]", cuda)


if __name__ == "__main__":
    unittest.main()
