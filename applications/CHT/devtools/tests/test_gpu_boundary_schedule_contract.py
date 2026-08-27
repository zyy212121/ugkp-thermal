import pathlib
import re
import unittest


TEST_FILE = pathlib.Path(__file__).resolve()
ROOT = TEST_FILE.parents[1]
if not (ROOT / "diluteUgkwpFoam.C").is_file():
    ROOT = TEST_FILE.parents[2]
CUDA = (
    ROOT / "private_backend" / "GpuResidentStrict.cu"
    if (ROOT / "private_backend" / "GpuResidentStrict.cu").is_file()
    else ROOT / "gpu" / "GpuResidentStrict.cu"
)


class GpuBoundaryScheduleContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = (ROOT / "diluteUgkwpFoam.C").read_text(encoding="utf-8")
        cls.header = (ROOT / "gpu" / "GpuResidentStrict.H").read_text(
            encoding="utf-8"
        )
        cls.api = (ROOT / "gpu" / "GpuBackendApi.H").read_text(
            encoding="utf-8"
        )
        cls.cuda = CUDA.read_text(encoding="utf-8")

    def test_schedule_is_configured_once_and_not_dispatched_per_step(self) -> None:
        self.assertIn("configureScheduledInlet", self.main)
        self.assertNotIn("dispatchScheduledBoundary", self.main)
        self.assertNotIn("updateScheduledInlet", self.main)

    def test_complete_tables_are_exposed_and_uploaded(self) -> None:
        schedule = (ROOT / "gpu" / "GpuBoundarySchedule.H").read_text(
            encoding="utf-8"
        )
        self.assertIn("copyColumns", schedule)
        self.assertIn("pressureTimes", self.main)
        self.assertIn("pressureValues", self.main)
        self.assertIn("volumeFractionTimes", self.main)
        self.assertIn("volumeFractionValues", self.main)
        self.assertIn("ConfigureScheduledInlet", self.api)

    def test_gpu_uses_clamped_linear_interpolation_for_both_tables(self) -> None:
        self.assertIn("linearScheduledValueDevice", self.cuda)
        self.assertRegex(
            self.cuda,
            r"scheduledPressureDevice\s*\([^)]*simulationTime",
        )
        self.assertRegex(
            self.cuda,
            r"scheduledSolidVolumeFractionDevice\s*\([^)]*simulationTime",
        )

    def test_old_per_step_update_abi_and_kernel_are_deleted(self) -> None:
        for source in (self.header, self.api, self.cuda):
            self.assertNotIn("UpdateScheduledInlet", source)
            self.assertNotIn("updateScheduledInletKernel", source)

    def test_advance_and_courant_paths_carry_schedule_time(self) -> None:
        self.assertRegex(
            self.header,
            r"computeGasCourant\s*\([^)]*scheduleTime",
        )
        self.assertRegex(
            self.api,
            r"ComputeGasCourant\s*\([^;]*scheduleTime",
        )
        self.assertRegex(
            self.api,
            r"AdvanceGasOnly\s*\([^;]*simulationTime",
        )
        self.assertIn("simulationTime", self.cuda)

    def test_post_scrub_configuration_publishes_only_schedule_metadata(self) -> None:
        self.assertIn("publishScheduledInletConfigurationKernel", self.cuda)
        configure = self.cuda.split(
            'extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet', 1
        )[1].split(
            'extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass', 1
        )[0]
        self.assertNotIn("syncDeviceState", configure)


if __name__ == "__main__":
    unittest.main()
