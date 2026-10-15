from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from source_contract_utils import read_source, function_block, braced_block


import re
import unittest
from pathlib import Path


UNIFIED_ROOT = SOURCE_ROOT
SHARED_WORKERS = (
    UNIFIED_ROOT / "gpu/thermal/CsrSegmentedPoolWorkers.cuh",
    UNIFIED_ROOT / "gpu/thermal/CsrSegmentedMomentWorkers.cuh",
)


BRANCHES = {
    "FSH": (
        UNIFIED_ROOT / "applications/FSH/gpu/GpuResidentStrict.H",
        UNIFIED_ROOT / "applications/FSH/private_backend/GpuResidentStrict.cu",
    ),
    "CHT": (
        UNIFIED_ROOT / "applications/CHT/gpu/GpuResidentStrict.H",
        UNIFIED_ROOT / "applications/CHT/gpu/GpuResidentStrict.cu",
    ),
}


class SegmentedTaskQueueFamilyContract(unittest.TestCase):
    @staticmethod
    def _braced_block(text: str, anchor: str) -> str:
        return braced_block(text,anchor)

    @staticmethod
    def _implementation_text(cuda_path: Path) -> str:
        return "\n".join(
            [read_source(cuda_path)]
            + [read_source(path) for path in SHARED_WORKERS]
        )

    def test_l2_requires_the_cell_local_cpcst_path(self) -> None:
        shared = read_source(UNIFIED_ROOT / "common/GpuSchedulingConfiguration.H")
        self.assertIn("value.csrCellLocalPath = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.warpAggregatedBinning = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.splitPreDirectory = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.csrLevel == GpuCsrLevel::L2", shared)
        for name, (header_path, _) in BRANCHES.items():
            header = read_source(header_path)
            self.assertIn("GpuSchedulingConfiguration", header, name)

    def test_l2_materializes_one_deterministic_queue_for_every_cell(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = self._implementation_text(cuda_path)
            for token in (
                "CsrReductionTaskSource",
                "countCsrReductionTasksKernel",
                "materializeCsrReductionTasksKernel",
                "csrCellTaskCount",
                "csrCellTaskOffset",
                "csrReductionTasks",
                "csrMultiTaskCellList",
            ):
                self.assertIn(token, cuda, f"{name}: {token}")
            self.assertIn("cub::DeviceScan::ExclusiveSum", cuda, name)
            self.assertRegex(
                cuda,
                r"csrCellTaskOffset,\s*s->nCells\s*\+\s*1",
                name,
            )

    def test_split_task_count_uses_one_logical_concatenation(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = self._implementation_text(cuda_path)
            self.assertIn("splitLogical", cuda, name)
            self.assertRegex(
                cuda,
                r"totalCount\s*=\s*baseCount\s*\+\s*injectionCount",
                name,
            )
            self.assertRegex(
                cuda,
                r"CsrReductionTaskSource::splitLogical",
                name,
            )
            self.assertIn("accumulateCsrSplitLogicalPoolTask", cuda, name)

    def test_enabled_path_is_not_light_then_heavy(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = self._implementation_text(cuda_path)
            for token in (
                "accumulateCsrSegmentedPoolTasksPersistentKernel",
                "accumulateCsrSegmentedMomentTasksPersistentKernel",
                "finalizeCsrSegmentedPoolCellsKernel",
                "finalizeCsrSegmentedMomentCellsKernel",
            ):
                self.assertIn(token, cuda, f"{name}: {token}")
            self.assertIn("csrCellTaskCount[c] == 1", cuda, name)
            self.assertIn("csrCellTaskCount[c] > 1", cuda, name)

    def test_l2_advance_does_not_launch_dead_light_reductions(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = read_source(cuda_path)
            advance = function_block(cuda,"ugkwpGpuResidentStrictAdvance")
            self.assertIsNotNone(advance, name)
            collision = advance.split("UGKP_DEV_PROBE_ENTER(ProbeCollisionPool);",1)[1].split("UGKP_DEV_PROBE_LEAVE(ProbeCollisionPool);",1)[0]
            stage = advance.split("UGKP_DEV_PROBE_ENTER(ProbeMoments);",1)[1].split("UGKP_DEV_PROBE_LEAVE(ProbeMoments);",1)[0]
            self.assertIn("launchPostTransportMomentPipeline",stage,name)
            moments = function_block(cuda,"launchPostTransportMomentPipeline")
            heavy_anchor = "if (s->csrHeavyReductionEnabled != 0)"
            collision_heavy = self._braced_block(collision, heavy_anchor)
            moments_heavy = self._braced_block(moments, heavy_anchor)
            self.assertIn("launchCsrHeavyPoolReduction", collision_heavy, name)
            self.assertNotIn("accumulatePoissonPool", collision_heavy, name)
            self.assertIn("launchCommonSegmentedMomentReduction", moments_heavy, name)
            self.assertNotIn("accumulateParticleMoments", moments_heavy, name)

    def test_pool_worker_selects_logical_or_single_source_once_per_task(self) -> None:
        pool = read_source(UNIFIED_ROOT / "gpu/thermal/CsrSegmentedPoolWorkers.cuh")
        worker = self._braced_block(
            pool, "struct CsrPoolOperation"
        )
        self.assertEqual(worker.count("accumulateCsrHeavyPoolTask<PoissonMode>"), 1)
        self.assertEqual(
            worker.count("accumulateCsrSplitLogicalPoolTask<PoissonMode>"), 1
        )
        self.assertIn("const bool directParticleIndex", worker)
        self.assertIn("GPU_POOL_SPLIT_DIRECTORY", worker)
        source = read_source(BRANCHES["FSH"][1])
        self.assertIn("#define GPU_POOL_SPLIT_DIRECTORY (descriptor.source == static_cast<int>(CsrReductionTaskSource::splitLogical))", source)

    def test_scheduler_protocol_is_shared_between_thermal_workers(self) -> None:
        shared = read_source(UNIFIED_ROOT / "common/CsrPersistentQueue.cuh")
        self.assertIn("atomicAdd(s.csrHeavyTaskCursor, 1)", shared)
        for path in SHARED_WORKERS:
            code = read_source(path)
            self.assertIn("runCsrPersistentQueue", code)
            self.assertIn("resetCsrPersistentQueue", code)
            self.assertNotIn("directThermalPoolDispatch", code)


    def test_l2_occupancy_queries_the_executed_segmented_workers(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = read_source(cuda_path)
            occupancy_name = (
                "configureParticleLaunchGeometry"
                if name == "gasUGKP"
                else "configureLaunchOccupancy"
            )
            occupancy = re.search(
                rf"int\s+{occupancy_name}\s*\([\s\S]*?\n\}}",
                cuda,
            )
            self.assertIsNotNone(occupancy, name)
            body = occupancy.group(0)
            self.assertIn("thermalPoolLaunchOccupancy", body, name)
            self.assertIn("accumulateCsrSegmentedMomentTasksPersistentKernel", body, name)
            self.assertNotIn("accumulateCsrHeavyPoolTasksPersistentKernel", body, name)
            self.assertNotIn("accumulateCsrHeavyMomentTasksPersistentKernel", body, name)

    def test_task_capacity_remains_a_conservative_logical_queue_bound(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = self._implementation_text(cuda_path)
            allocation = re.search(
                r"const size_t segmentedTaskCapacity\s*=(?P<body>[\s\S]*?);",
                cuda,
            )
            self.assertIsNotNone(allocation, name)
            self.assertRegex(allocation.group("body"), r"2u\s*\*\s*nc", name)
            self.assertIn("segmented task capacity exceeds 32-bit indexing", cuda, name)

    def test_split_capacity_bound_and_interval_partition(self) -> None:
        for block in (32, 64, 128, 256):
            for base in (
                (0, 1, block - 1, block, block + 1),
                (3, 17, 257, 1025, 4097),
            ):
                injection = tuple(reversed(base))
                tasks = 0
                population = 0
                for base_count, injection_count in zip(base, injection):
                    population += base_count + injection_count
                    total_count = base_count + injection_count
                    tasks += (total_count + block - 1) // block
                    logical_lengths = [
                        min(block, total_count - begin)
                        for begin in range(0, total_count, block)
                    ]
                    self.assertEqual(sum(logical_lengths), total_count)
                    self.assertTrue(
                        all(0 < length <= block for length in logical_lengths)
                    )
                capacity = (population + block - 1) // block + 2 * len(base) + 1
                self.assertLessEqual(tasks, capacity)

    def test_small_injection_tail_does_not_create_a_second_task(self) -> None:
        block = 256
        base_count = 100
        injection_count = 1
        total_count = base_count + injection_count
        self.assertEqual((total_count + block - 1) // block, 1)


if __name__ == "__main__":
    unittest.main()
