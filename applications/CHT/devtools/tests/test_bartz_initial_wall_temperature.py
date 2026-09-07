import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "mss7_draw", ROOT / "examples/thermal/MSS7_turbulent_wallModel/draw.py"
)
DRAW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRAW)


class InitialWallTemperatureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        for region, patch in [("fluid", "fluid_to_graphite"), ("graphite", "graphite_to_fluid")]:
            mesh = self.case / "constant" / region / "polyMesh"
            mesh.mkdir(parents=True)
            (mesh / "boundary").write_text(patch + " { nFaces 2; startFace 0; }")
            (mesh / "points").write_text("6\n(\n(0 0 0) (1 0 0) (0 1 0) (2 0 0) (3 0 0) (2 1 0)\n)\n")
            (mesh / "faces").write_text("2\n(\n3(0 1 2) 3(3 4 5)\n)\n")
            (mesh / "owner").write_text("2\n(\n1 0\n)\n")
        (self.case / "constant/solidRegionProperties").write_text(
            "solidThermalCoupling { mapping oneToOneConformal; solidRegion graphite; "
            "coupledInterfaces ( { fluidPatch fluid_to_graphite; solidPatch graphite_to_fluid; } ); }"
        )
        for region in ["fluid", "graphite"]:
            (self.case / "1" / region).mkdir(parents=True)
        (self.case / "1/thermalExchangeState").write_text("initialState true;")
        (self.case / "1/fluid/T").write_text("fluid_to_graphite { value uniform 893.836; }")
        (self.case / "1/graphite/T").write_text("internalField nonuniform List<scalar> 2 (700 300);")

    def read(self):
        return DRAW.interval_wall_temperature(self.case, self.case / "1", "fluid_to_graphite", 2)

    def test_initial_uses_solid_owner_order_and_ignores_fluid_placeholder(self):
        values, source = self.read()
        np.testing.assert_array_equal(values, [300, 700])
        self.assertEqual(source, "initial_solid_owner_mapped")

    def test_uniform_initial_solid(self):
        (self.case / "1/graphite/T").write_text("internalField uniform 400;")
        np.testing.assert_array_equal(self.read()[0], [400, 400])

    def test_completed_state_uses_uploaded_fluid_temperature(self):
        (self.case / "1/thermalExchangeState").write_text("initialState false;")
        (self.case / "1/fluid/T").write_text("fluid_to_graphite { value nonuniform List<scalar> 2 (450 800); }")
        values, source = self.read()
        np.testing.assert_array_equal(values, [450, 800])
        self.assertEqual(source, "previous_fluid_boundary")

    def test_missing_solid_is_not_replaced_by_fluid_placeholder(self):
        (self.case / "1/graphite/T").unlink()
        with self.assertRaises(FileNotFoundError):
            self.read()

    def test_reordered_geometry_rejected(self):
        (self.case / "constant/graphite/polyMesh/faces").write_text("2\n(\n3(3 4 5) 3(0 1 2)\n)\n")
        with self.assertRaisesRegex(RuntimeError, "solver order"):
            self.read()

    def test_bartz_responds_to_actual_wall_temperature(self):
        import json
        c = json.loads((ROOT / "examples/thermal/MSS7_turbulent_wallModel/assets/postprocessing/bartz.json").read_text())
        def flux(tw):
            return DRAW.bartz_wall_heat_flux(3e6, tw, 0.064938134403, 0.0015226, 0.152405,
                c["total_temperature_k"], c["throat_diameter_m"], c["throat_curvature_radius_m"],
                c["dynamic_viscosity_pa_s"], c["specific_heat_j_kg_k"], c["prandtl"],
                c["gamma"], c["gas_constant_j_kg_k"])[0]
        self.assertGreater(flux(300), flux(893.836))


if __name__ == "__main__":
    unittest.main()
