from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "common/GpuSchedulingConfiguration.H"
FORBIDDEN_KEYS = (
    "gpuResidentStrict",
    "gpuCsrCellLocalPath",
    "gpuCsrHeavyReduction",
    "gpuCsrWarpAggregatedBinning",
    "gpuCsrSplitPreDirectory",
)


class StrictCsrLevelContract(unittest.TestCase):
    def test_one_public_level_derives_every_internal_path_flag(self) -> None:
        text = SHARED.read_text(encoding="utf-8")
        self.assertIn('"gpuCsrLevel"', text)
        for level in ("L0", "L1", "L2", "auto"):
            self.assertRegex(text, rf'csrLevelValue\s*==\s*"{level}"')
        self.assertRegex(
            text,
            r"csrCellLocalPath\s*=\s*value\.csrLevel\s*!=\s*GpuCsrLevel::L0",
        )
        self.assertRegex(
            text,
            r"warpAggregatedBinning\s*=\s*value\.csrLevel\s*!=\s*GpuCsrLevel::L0",
        )
        self.assertRegex(
            text,
            r"splitPreDirectory\s*=\s*value\.csrLevel\s*!=\s*GpuCsrLevel::L0",
        )
        self.assertIn("GpuCsrLevel::L2", text)
        self.assertIn("GpuCsrLevel::automatic", text)

    def test_removed_public_switches_are_rejected_not_silently_ignored(self) -> None:
        text = SHARED.read_text(encoding="utf-8")
        self.assertIn("forbiddenGpuSchedulingKeys", text)
        for key in FORBIDDEN_KEYS:
            self.assertIn(f'"{key}"', text)
        self.assertIn("has been removed", text)
        self.assertIn("gpuCsrLevel L0|L1|L2|auto", text)

    def test_all_examples_use_exactly_one_level_and_no_removed_switches(self) -> None:
        schedules = sorted((ROOT / "examples").glob("**/constant/schedulingProperties"))
        self.assertTrue(schedules)
        for path in schedules:
            text = path.read_text(encoding="utf-8")
            levels = re.findall(r"^\s*gpuCsrLevel\s+(L0|L1|L2|auto)\s*;", text, re.M)
            self.assertEqual(len(levels), 1, str(path))
            for key in FORBIDDEN_KEYS:
                self.assertNotRegex(text, rf"^\s*{key}\b", str(path))
            if levels[0] == "auto":
                self.assertRegex(
                    text,
                    r"^\s*gpuCsrHeavyReductionAutoInterval\s+[1-9][0-9]*\s*;",
                )
            else:
                self.assertNotRegex(
                    text,
                    r"^\s*gpuCsrHeavyReductionAutoInterval\b",
                    str(path),
                )

    def test_three_solver_entry_points_do_not_read_resident_strict(self) -> None:
        for branch in ("gasUGKP", "FSH", "CHT"):
            source = (ROOT / f"applications/{branch}/diluteUgkwpFoam.C").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("gpuScheduling.residentStrict", source, branch)
            self.assertNotRegex(
                source,
                r"lookup(?:OrDefault)?[^\n]*gpuResidentStrict",
                branch,
            )


if __name__ == "__main__":
    unittest.main()
