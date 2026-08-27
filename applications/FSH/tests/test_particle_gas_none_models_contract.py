import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "examples" / "thermal" / "singleAluminaDrop"


class ParticleGasNoneModelsContractTests(unittest.TestCase):
    def test_all_three_solvers_implement_drag_none(self):
        for solver in ("FSH", "CHT", "gasUGKP"):
            source = (
                ROOT / "applications" / solver / "gpu" / "GpuDragModel.H"
            ).read_text()
            self.assertIn('== "none"', source, solver)

    def test_all_three_solvers_implement_heat_transfer_none(self):
        for solver in ("FSH", "CHT", "gasUGKP"):
            configuration = (
                (
                    ROOT / "applications" / solver / "diluteUgkwpFoam.C"
                ).read_text()
                + (
                    ROOT / "applications" / solver / "createFields.H"
                ).read_text()
            )
            cuda_subdir = "private_backend" if solver != "CHT" else "gpu"
            cuda = (
                ROOT
                / "applications"
                / solver
                / cuda_subdir
                / "GpuResidentStrict.cu"
            ).read_text()
            self.assertIn("particleGasHeatTransferModel", configuration, solver)
            self.assertIn("particleGasHeatTransferModelId", cuda, solver)

    def test_case_uses_formal_models_and_public_solver(self):
        properties = (CASE / "constant" / "particleProperties").read_text()
        run = (CASE / "run_variant").read_text()
        self.assertIn("dragModel none;", properties)
        self.assertIn("particleGasHeatTransferModel none;", properties)
        self.assertIn('"${FOAM_USER_APPBIN}/FSH"', run)
        self.assertNotIn("singleDropFSHCudaBackend", run)
        self.assertNotIn("private_solver", run)

    def test_compile_time_single_drop_bypass_is_removed(self):
        for solver in ("FSH", "gasUGKP"):
            cuda = (
                ROOT
                / "applications"
                / solver
                / "private_backend"
                / "GpuResidentStrict.cu"
            ).read_text()
            self.assertNotIn("UGKP_SINGLE_DROP_ISOLATED_COUPLING", cuda)


if __name__ == "__main__":
    unittest.main()
