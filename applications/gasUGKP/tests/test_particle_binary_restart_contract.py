#!/usr/bin/env python3
                                                                           

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "gpu" / "GpuResidentStrict.H"


class ParticleBinaryRestartContract(unittest.TestCase):
    def test_v4_v5_remain_supported_and_ugkp_format_is_additive(self) -> None:
        source = FRONTEND.read_text()
        self.assertIn('restartHeader == "UGKP_PARTICLES_SCHEMA4"', source)
        self.assertIn('restartHeader == "UGKP_PARTICLES_SCHEMA5_BIN"', source)
        self.assertIn('restartHeader == "UGKP_PARTICLES_SCHEMA1_BIN"', source)

    def test_binary_reader_is_chunked_and_width_checked(self) -> None:
        source = FRONTEND.read_text()
        required = (
            "readBinaryParticleBlock",
            "binary restart scalar width mismatch",
            "binary restart label width mismatch",
            "binary restart unsigned width mismatch",
            "binary restart chunk count",
            "binary restart trailing data",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_v5_reuses_the_established_upload_contract(self) -> None:
        source = FRONTEND.read_text()
        self.assertIn("ugkwpGpuResidentStrictUploadParticleRestartMirror", source)
        self.assertNotIn("ugkwpGpuResidentStrictUploadBinaryParticleRestart", source)

    def test_ugkp_binary_restart_persists_particle_mass(self) -> None:
        source = FRONTEND.read_text()
        self.assertIn('os << "UGKP_PARTICLES_SCHEMA1_BIN "', source)
        self.assertIn(
            'writeBinaryParticleBlock(os, pm.begin() + offset, count, "pm", restartFile);',
            source,
        )
        self.assertIn(
            'readBinaryParticleBlock(is, pm.begin() + offset, count, "pm", restartFile);',
            source,
        )

    def test_restart_upload_never_forces_existing_mass_to_injection_mass(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        self.assertNotIn("forceParticleMassToParcelMassKernel", cuda)
        self.assertNotIn("s.pm[i] = s.injectionParcelMass", cuda)

    def test_legacy_mass_and_future_injection_mass_are_distinct(self) -> None:
        wrapper = FRONTEND.read_text()
        fields = (ROOT / "createFields.H").read_text()
        self.assertIn("legacyRestartParcelMass_", wrapper)
        self.assertIn('"injectionParcelMass"', fields)
        self.assertIn('"legacyRestartParcelMass"', fields)


if __name__ == "__main__":
    unittest.main()
