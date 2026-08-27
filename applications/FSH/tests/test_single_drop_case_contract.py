import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "examples" / "thermal" / "singleAluminaDrop"


class CaseContractTests(unittest.TestCase):
    def test_mesh_and_wall_capacity(self):
        mesh = (CASE / "system" / "blockMeshDict").read_text()
        self.assertIn("(1 1 10)", mesh)
        self.assertRegex(mesh, r"impactWall\s*\{\s*type wall;")
        face_area = 0.04 * 0.04
        maximum_area = 4.2507311368088e-4
        self.assertGreater(0.63 * face_area, maximum_area)

    def test_one_physical_drop(self):
        properties = (CASE / "constant" / "particleProperties").read_text()
        density = float(re.search(r"\brhoS\s+([0-9.eE+-]+);", properties).group(1))
        parcel_mass = float(re.search(r"\bparcelMass\s+([0-9.eE+-]+);", properties).group(1))
        expected_mass = math.pi * density * 0.005 ** 3 / 6.0
        self.assertAlmostEqual(parcel_mass, expected_mass, places=15)
        self.assertIn("contactAngleDegree 70;", properties)
        self.assertIn("maximumCoverage 0.63;", properties)
        restart = (CASE / "0" / "gpuResidentStrictParticles.dat").read_text().splitlines()
        self.assertEqual(restart[0], "UGKP_PARTICLES_SCHEMA3 1")
        values = restart[1].split()
        self.assertEqual(values[0:6], ["0.02", "0.02", "0.0005", "0", "0", "-10"])
        self.assertEqual(values[6], "2373.15")
        self.assertEqual(values[8], "0.005")
        self.assertEqual(values[13:16], ["0", "-1", "0"])

    def test_runtime_scope_is_explicit(self):
        scheduling = (CASE / "constant" / "schedulingProperties").read_text()
        control = (CASE / "system" / "controlDict").read_text()
        self.assertIn("gpuResidentPureGasOnly false;", scheduling)
        self.assertIn("gpuResidentDynamicInlet false;", scheduling)
        self.assertIn("gpuCsrLevel L0;", scheduling)
        self.assertIn("endTime 0.010051;", control)
        self.assertIn("writeInterval 5e-5;", control)

    def test_formal_decoupling_models_are_explicit(self):
        properties = (CASE / "constant" / "particleProperties").read_text()
        self.assertIn("dragModel none;", properties)
        self.assertIn("particleGasHeatTransferModel none;", properties)

    def test_cold_wall_case_is_runnable(self):
        expected = {"coldWall": "wallInteractionModel coldWall1D;"}
        allrun = (CASE / "Allrun").read_text()
        allclean = (CASE / "Allclean").read_text()
        for variant, control in expected.items():
            properties = CASE / variant / "constant" / "particleProperties"
            self.assertTrue(properties.is_file())
            self.assertIn(control, properties.read_text())
            self.assertIn(f'"${{case_root}}/run_variant" {variant}', allrun)
            self.assertIn(f'"${{case_root}}/clean_variant" {variant}', allclean)
        self.assertFalse((CASE / "coldWall2d").exists())


if __name__ == "__main__":
    unittest.main()
