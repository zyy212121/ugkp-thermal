import tempfile
import unittest
from pathlib import Path

from devtools.generate_cht_perf_particles import (
    allocate_counts,
    read_cell_centres,
    read_cell_volumes,
    read_source_faces,
    splitmix64,
    validate_parcel_mass,
    write_epsg_prev,
    write_restart,
    write_zero_source_residual,
)


class PerformanceParticleGeneratorTests(unittest.TestCase):
    def test_allocate_counts_is_cell_sorted_and_balanced(self):
        self.assertEqual(allocate_counts(4, 10), [3, 3, 2, 2])

    def test_splitmix_is_deterministic_and_nonzero(self):
        self.assertEqual(splitmix64(17), splitmix64(17))
        self.assertNotEqual(splitmix64(17), 0)
        self.assertNotEqual(splitmix64(17), splitmix64(18))

    def test_parcel_mass_must_be_strictly_below_limit(self):
        validate_parcel_mass(5e-9)
        with self.assertRaises(ValueError):
            validate_parcel_mass(5e-8)
        with self.assertRaises(ValueError):
            validate_parcel_mass(float("nan"))

    def test_read_and_write_restart(self):
        field_text = """FoamFile\n{\n}\ninternalField nonuniform List<vector>\n2\n(\n(1 2 3)\n(4 5 6)\n)\n;\n"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            centres_path = root / "C"
            restart_path = root / "gpuResidentStrictParticles.dat"
            centres_path.write_text(field_text, encoding="utf-8")

            centres = read_cell_centres(centres_path)
            write_restart(restart_path, centres, 5)

            rows = restart_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "UGKP_PARTICLES_SCHEMA3 5")
            self.assertEqual(len(rows), 6)
            self.assertEqual([int(row.split()[9]) for row in rows[1:]], [0, 0, 0, 1, 1])
            self.assertEqual([int(row.split()[12]) for row in rows[1:]], [1, 2, 3, 4, 5])
            self.assertTrue(all(len(row.split()) == 15 for row in rows[1:]))

    def test_read_volumes_and_write_consistent_auxiliary_mirrors(self):
        field_text = """FoamFile\n{\n}\ninternalField nonuniform List<scalar>\n2\n(\n1e-8\n2e-8\n)\n;\n"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volumes_path = root / "V"
            epsg_path = root / "gpuResidentStrictEpsGPrev.dat"
            residual_template_path = root / "source-template.dat"
            residual_path = root / "gpuResidentStrictSourceResidual.dat"
            volumes_path.write_text(field_text, encoding="utf-8")
            residual_template_path.write_text(
                "UGKP_SOURCE_RESIDUAL_SCHEMA1 2\n17 1e-12\n23 2e-12\n",
                encoding="utf-8",
            )

            volumes = read_cell_volumes(volumes_path)
            write_epsg_prev(epsg_path, volumes, [2, 1], 5e-9, 2800.0)
            source_faces = read_source_faces(residual_template_path)
            write_zero_source_residual(residual_path, source_faces)

            epsg = epsg_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(epsg[0], "2")
            self.assertAlmostEqual(float(epsg[1]), 1 - 1e-8 / 2800.0 / 1e-8)
            self.assertAlmostEqual(float(epsg[2]), 1 - 5e-9 / 2800.0 / 2e-8)
            self.assertEqual(
                residual_path.read_text(encoding="utf-8"),
                "UGKP_SOURCE_RESIDUAL_SCHEMA1 2\n"
                "17 0.00000000000000000e+00\n"
                "23 0.00000000000000000e+00\n",
            )


if __name__ == "__main__":
    unittest.main()
