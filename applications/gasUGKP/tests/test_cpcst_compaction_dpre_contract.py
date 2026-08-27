#!/usr/bin/env python3
                                                                             

from pathlib import Path
import re
import unittest

from source_contract_utils import function_block


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "private_backend" / "GpuResidentStrict.cu").read_text(
    encoding="utf-8", errors="strict"
)


class CompactionGeneratedDpreContract(unittest.TestCase):
    def test_device_state_keeps_a_compacted_base_directory(self) -> None:
        for symbol in (
            "preBaseCellOffset",
            "preBaseParticleCountDevice",
            "preBaseDirectoryReady",
        ):
            self.assertIn(symbol, CUDA)

        release = function_block(CUDA, "releaseState")
        allocate = function_block(CUDA, "allocateFields")
        for symbol in ("preBaseCellOffset", "preBaseParticleCountDevice"):
            self.assertIn(f"release(s->{symbol})", release)
            self.assertIn(f"s->{symbol}", allocate)

    def test_preparation_bins_only_the_appended_injection_range(self) -> None:
        prepare = function_block(CUDA, "preparePreTransportParticleDirectory")
        count = function_block(CUDA, "countInjectedParticlesByCellKernel")
        scatter = function_block(CUDA, "scatterInjectedParticlesByCellKernel")

        self.assertIn("preBaseDirectoryReady", prepare)
        self.assertIn("binParticlesByCell", prepare)
        preparation_body = prepare[prepare.index("const int cellGrid") :]
        fallback_body = prepare[: prepare.index("const int cellGrid")]
        self.assertIn("if (s->preBaseDirectoryReady == 0)", fallback_body)
        self.assertNotIn("csrHeavyReductionEnabled", fallback_body)
        self.assertEqual(fallback_body.count("binParticlesByCell"), 2)
        self.assertIn("csrSplitPreDirectoryEnabled", fallback_body)
        self.assertNotIn("binParticlesByCell", preparation_body)
        for body in (count, scatter):
            self.assertIn("preBaseParticleCountDevice", body)
            self.assertIn("particleCountDevice", body)
        self.assertIn("sortedParticleIndex", scatter)

    def test_injection_tail_honours_warp_aggregated_binning(self) -> None:
        prepare = function_block(CUDA, "preparePreTransportParticleDirectory")
        count = function_block(CUDA, "countInjectedParticlesByCellKernel")
        scatter = function_block(CUDA, "scatterInjectedParticlesByCellKernel")

        for body in (count, scatter):
            self.assertIn("WarpAggregated", body)
            self.assertIn("__ballot_sync", body)
            self.assertIn("__match_any_sync", body)
        self.assertIn("csrWarpAggregatedBinning", prepare)
        self.assertIn("countInjectedParticlesByCellKernel<true>", prepare)
        self.assertIn("countInjectedParticlesByCellKernel<false>", prepare)
        self.assertIn("scatterInjectedParticlesByCellKernel<true>", prepare)
        self.assertIn("scatterInjectedParticlesByCellKernel<false>", prepare)

    def test_source_free_split_pre_skips_empty_injection_work(self) -> None:
        prepare = function_block(CUDA, "preparePreTransportParticleDirectory")
        source_free = function_block(CUDA, "prepareSourceFreeSplitPreDirectory")
        split_light = function_block(
            CUDA, "launchSplitPrePoissonPoolLightReduction"
        )
        split_heavy = function_block(CUDA, "launchCsrHeavyPoolReduction")

        self.assertIn("if (s->nBoundarySources == 0)", prepare)
        self.assertIn("prepareSourceFreeSplitPreDirectory", prepare)
        self.assertIn("cellParticleOffset", source_free)
        self.assertIn("preInjectionSegmentActive = 0", source_free)
        self.assertIn("preInjectionSegmentActive = 1", prepare)
        self.assertIn("if (s->preInjectionSegmentActive != 0)", split_light)
        self.assertIn("if (s->preInjectionSegmentActive != 0)", split_heavy)

    def test_heavy_preparation_preserves_base_and_injection_segments(self) -> None:
        prepare = function_block(CUDA, "preparePreTransportParticleDirectory")
        self.assertIn("prepareSplitPreCsrHeavyReductionTasks", prepare)

        split_tasks = function_block(
            CUDA, "prepareSplitPreCsrHeavyReductionTasks"
        )
        self.assertNotIn("binParticlesByCell", split_tasks)
        self.assertIn("prepareCsrSegmentedReductionTasks", split_tasks)
        count_tasks = function_block(CUDA, "countCsrReductionTasksKernel")
        self.assertIn("preBaseCellOffset", count_tasks)
        self.assertIn("cellParticleOffset", count_tasks)
        self.assertIn("baseTasks + injectionTasks", count_tasks)
        materialize = function_block(CUDA, "materializeCsrReductionTasksKernel")
        self.assertIn("splitBaseDirect", materialize)
        self.assertIn("splitInjectionIndexed", materialize)

        heavy_pool = function_block(CUDA, "launchCsrHeavyPoolReduction")
        self.assertIn("launchCsrSegmentedPoolReduction", heavy_pool)
        heavy_task = function_block(CUDA, "accumulateCsrHeavyPoolTask")
        self.assertIn("const bool directParticleIndex", heavy_task)
        self.assertIn("directParticleIndex", heavy_task)
        self.assertIn("s.sortedParticleIndex[pos]", heavy_task)

        light_pool = function_block(
            CUDA, "accumulatePoissonPoolSplitSegmentByCellKernel"
        )
        self.assertIn("HeavyReductionEnabled", light_pool)
        self.assertIn("if constexpr (HeavyReductionEnabled)", light_pool)
        self.assertRegex(
            light_pool,
            r"if constexpr \(HeavyReductionEnabled\)\s*\{\s*return;\s*\}",
        )
        self.assertIn("preBaseCellOffset", light_pool)
        self.assertIn("cellParticleOffset", light_pool)

    def test_compaction_builds_the_next_base_heavy_schedule(self) -> None:
        advance = function_block(CUDA, "ugkwpGpuResidentStrictAdvance")
        capture_pos = advance.index("captureCompactedPreBaseOffsetsKernel")
        heavy_pos = advance.index("prepareCsrHeavyBaseReductionTasks")
        self.assertLess(capture_pos, heavy_pos)

    def test_pre_consumers_traverse_base_and_injection_segments(self) -> None:
        pressure = function_block(CUDA, "applyCollisionalPressureProjectionKernel")
        self.assertIn("SplitDirectory", pressure)
        self.assertNotIn("const int useSplitPreDirectory", pressure)
        self.assertIn("preBaseCellOffset", pressure)
        self.assertIn("cellParticleOffset", pressure)
        self.assertIn("sortedParticleIndex", pressure)
        pressure_declaration = re.compile(
            r"template\s*<\s*bool\s+SplitDirectory\s*>\s*"
            r"__global__\s+void\s+applyCollisionalPressureProjectionKernel\s*\("
        )
        self.assertRegex(CUDA, pressure_declaration)

        split_poisson = function_block(
            CUDA, "accumulatePoissonPoolSplitSegmentByCellKernel"
        )
        for token in (
            "DirectParticleIndex",
            "AddToExistingPool",
            "HeavyReductionEnabled",
            "preBaseCellOffset",
            "cellParticleOffset",
            "sortedParticleIndex",
        ):
            self.assertIn(token, split_poisson)

        pressure_launch = function_block(CUDA, "applyCollisionalPressureKick")
        advance = function_block(CUDA, "ugkwpGpuResidentStrictAdvance")
        self.assertIn("applyCollisionalPressureProjectionKernel<true>", pressure_launch)
        self.assertIn("applyCollisionalPressureProjectionKernel<false>", pressure_launch)
        self.assertIn("launchSplitPrePoissonPoolLightReduction", advance)
        self.assertIn("accumulatePoissonPoolParticlesByCellKernel", advance)
        split_launch = function_block(
            CUDA, "launchSplitPrePoissonPoolLightReduction"
        )
        self.assertIn(
            "accumulatePoissonPoolSplitSegmentByCellKernel<true, false, false>",
            split_launch,
        )
        self.assertIn(
            "accumulatePoissonPoolSplitSegmentByCellKernel<false, true, false>",
            split_launch,
        )

    def test_compaction_captures_the_next_base_but_post_bin_stays_full(self) -> None:
        capture = function_block(CUDA, "captureCompactedPreBaseOffsetsKernel")
        self.assertIn("compactCellOffset", capture)
        self.assertIn("preBaseCellOffset", capture)

        advance = function_block(CUDA, "ugkwpGpuResidentStrictAdvance")
        self.assertIn("preparePreTransportParticleDirectory", advance)
        self.assertIn("captureCompactedPreBaseOffsetsKernel", advance)
        post_marker = advance.index("UGKP_DEV_PROBE_ENTER(ProbeBinPost)")
        post_region = advance[post_marker : post_marker + 600]
        self.assertIn("binParticlesByCell", post_region)
        full_bin = function_block(CUDA, "binParticlesByCell")
        self.assertIn("countParticlesByCellKernel", full_bin)
        self.assertIn("scatterParticlesByCellKernel", full_bin)
        self.assertIn("prepareCsrHeavyReductionTasks", full_bin)

    def test_probe_csv_header_matches_the_thirteen_emitted_stages(self) -> None:
        header = function_block(CUDA, "writeDevelopmentProbeHeader")
        self.assertNotIn("theta_pool_ms", header)
        expected = (
            "total_ms,gas_flux_ms,eulerian_coupling_ms,injection_ms,bin_pre_ms,"
            "pressure_pre_ms,collision_pool_ms,relax_ms,track_ms,bin_post_ms,"
            "moments_ms,pressure_post_ms,compaction_ms,boundary_ms"
        )
        self.assertIn(expected, header.replace('"\n        "', ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
