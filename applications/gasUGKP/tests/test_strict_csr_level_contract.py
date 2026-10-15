from __future__ import annotations
from source_contract_utils import read_source


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
    def test_all_examples_use_exactly_one_level_and_no_removed_switches(self) -> None:
        schedules = sorted((ROOT / "examples").glob("**/constant/schedulingProperties"))
        self.assertTrue(schedules)
        for path in schedules:
            text = read_source(path)
            levels = re.findall(r"^\s*gpuCsrLevel\s+(L0|L1|L2|auto)\s*;", text, re.M)
            self.assertEqual(len(levels), 1, str(path))
            for key in FORBIDDEN_KEYS:
                self.assertNotRegex(text, rf"^\s*{key}\b", str(path))
            self.assertNotRegex(text, r"(?m)^\s*gpuResearchVariant\b", str(path))
            if levels[0] == "auto":
                self.assertRegex(text, r"(?m)^\s*gpuCsrHeavyReductionAutoInterval\s+[1-9][0-9]*\s*;", str(path))

    def test_three_solver_entry_points_do_not_read_resident_strict(self) -> None:
        for branch in ("gasUGKP", "FSH", "CHT"):
            source = read_source(ROOT / f"applications/{branch}/diluteUgkwpFoam.C")
            self.assertNotIn("gpuScheduling.residentStrict", source, branch)
            self.assertNotRegex(
                source,
                r"lookup(?:OrDefault)?[^\n]*gpuResidentStrict",
                branch,
            )


if __name__ == "__main__":
    unittest.main()
