from __future__ import annotations

import re
import unittest
from pathlib import Path


UNIFIED_ROOT = Path(__file__).resolve().parents[3]
SHARED_WORKERS = (
    UNIFIED_ROOT / "gpu/CsrSegmentedPoolWorkers.cuh",
    UNIFIED_ROOT / "gpu/CsrSegmentedMomentWorkers.cuh",
)


BRANCHES = {
    "gasUGKP": (
        UNIFIED_ROOT / "applications/gasUGKP/gpu/GpuResidentStrict.H",
        UNIFIED_ROOT / "applications/gasUGKP/private_backend/GpuResidentStrict.cu",
    ),
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
        start = text.index(anchor)
        opening = text.index("{", start)
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1 : index]
        raise AssertionError(f"unterminated block after {anchor}")

    @staticmethod
    def _implementation_text(cuda_path: Path) -> str:
        return "\n".join(
            [cuda_path.read_text(encoding="utf-8")]
            + [path.read_text(encoding="utf-8") for path in SHARED_WORKERS]
        )

    def test_l2_requires_the_cell_local_cpcst_path(self) -> None:
        shared = (
            UNIFIED_ROOT / "common/GpuSchedulingConfiguration.H"
        ).read_text(encoding="utf-8")
        self.assertIn("value.csrCellLocalPath = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.warpAggregatedBinning = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.splitPreDirectory = value.csrLevel != GpuCsrLevel::L0", shared)
        self.assertIn("value.csrLevel == GpuCsrLevel::L2", shared)
        for name, (header_path, _) in BRANCHES.items():
            header = header_path.read_text(encoding="utf-8")
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

    def test_split_task_count_keeps_direct_and_indexed_sources_separate(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = self._implementation_text(cuda_path)
            self.assertIn("splitBaseDirect", cuda, name)
            self.assertIn("splitInjectionIndexed", cuda, name)
            self.assertRegex(
                cuda,
                r"baseTasks\s*\+\s*injectionTasks",
                name,
            )
            self.assertRegex(
                cuda,
                r"source\s*==\s*static_cast<int>\s*\(\s*CsrReductionTaskSource::splitBaseDirect",
                name,
            )

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
            cuda = cuda_path.read_text(encoding="utf-8")
            advance = re.search(
                r"int\s+ugkwpGpuResidentStrictAdvance\s*\([\s\S]*?\n\}",
                cuda,
            )
            self.assertIsNotNone(advance, name)
            collision = advance.group(0).split(
                "UGKP_DEV_PROBE_ENTER(ProbeCollisionPool);", 1
            )[1].split("UGKP_DEV_PROBE_LEAVE(ProbeCollisionPool);", 1)[0]
            moments = advance.group(0).split(
                "UGKP_DEV_PROBE_ENTER(ProbeMoments);", 1
            )[1].split("UGKP_DEV_PROBE_LEAVE(ProbeMoments);", 1)[0]
            heavy_anchor = "if (s->csrHeavyReductionEnabled != 0)"
            collision_heavy = self._braced_block(collision, heavy_anchor)
            moments_heavy = self._braced_block(moments, heavy_anchor)
            self.assertIn("launchCsrHeavyPoolReduction", collision_heavy, name)
            self.assertNotIn("accumulatePoissonPool", collision_heavy, name)
            self.assertIn("launchCsrHeavyMomentReduction", moments_heavy, name)
            self.assertNotIn("accumulateParticleMoments", moments_heavy, name)

    def test_zero_collision_probability_skips_eight_component_reduction(self) -> None:
        pool = BRANCHES["gasUGKP"][1].read_text(encoding="utf-8")
        body = self._braced_block(
            pool, "if (PoissonMode && collisionProbability <= 0.0)"
        )
        self.assertNotIn("blockReduceComponentSums", body)
        self.assertIn("csrHeavyPartials", body)
        self.assertIn("continue", body)

    def test_pool_worker_has_one_runtime_source_selection_not_two_inlined_tasks(self) -> None:
        pool = (UNIFIED_ROOT / "gpu/CsrSegmentedPoolWorkers.cuh").read_text(
            encoding="utf-8"
        )
        worker = self._braced_block(
            pool, "__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel"
        )
        self.assertEqual(worker.count("accumulateCsrHeavyPoolTask<PoissonMode>"), 1)
        self.assertIn("const bool directParticleIndex", worker)
        self.assertIn("descriptor.source", worker)

    def test_q1_workers_publish_shared_descriptor_with_one_scheduler_barrier(self) -> None:
        implementations = (
            (
                "gas pool",
                BRANCHES["gasUGKP"][1],
                "accumulateCsrSegmentedPoolTasksPersistentKernel",
            ),
            (
                "shared pool",
                UNIFIED_ROOT / "gpu/CsrSegmentedPoolWorkers.cuh",
                "accumulateCsrSegmentedPoolTasksPersistentKernel",
            ),
            (
                "gas moments",
                BRANCHES["gasUGKP"][1],
                "accumulateCsrSegmentedMomentTasksPersistentKernel",
            ),
            (
                "shared moments",
                UNIFIED_ROOT / "gpu/CsrSegmentedMomentWorkers.cuh",
                "accumulateCsrSegmentedMomentTasksPersistentKernel",
            ),
        )
        for label, path, anchor in implementations:
            worker = self._braced_block(path.read_text(encoding="utf-8"), anchor)
            self.assertNotIn("taskBatchSize", worker, label)
            self.assertIn("atomicAdd(s.csrHeavyTaskCursor, 1)", worker, label)
            self.assertIn("__shared__ CsrReductionTask descriptor", worker, label)
            self.assertIn("descriptor = s.csrReductionTasks[task]", worker, label)
            self.assertEqual(worker.count("__syncthreads()"), 1, label)

    def test_l2_occupancy_queries_the_executed_segmented_workers(self) -> None:
        for name, (_, cuda_path) in BRANCHES.items():
            cuda = cuda_path.read_text(encoding="utf-8")
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
            self.assertIn("accumulateCsrSegmentedPoolTasksPersistentKernel", body, name)
            self.assertIn("accumulateCsrSegmentedMomentTasksPersistentKernel", body, name)
            self.assertNotIn("accumulateCsrHeavyPoolTasksPersistentKernel", body, name)
            self.assertNotIn("accumulateCsrHeavyMomentTasksPersistentKernel", body, name)

    def test_task_capacity_covers_two_split_source_boundaries_per_cell(self) -> None:
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
                    base_tasks = (base_count + block - 1) // block
                    injection_tasks = (injection_count + block - 1) // block
                    tasks += base_tasks + injection_tasks
                    base_lengths = [
                        min(block, base_count - begin)
                        for begin in range(0, base_count, block)
                    ]
                    injection_lengths = [
                        min(block, injection_count - begin)
                        for begin in range(0, injection_count, block)
                    ]
                    self.assertEqual(sum(base_lengths), base_count)
                    self.assertEqual(sum(injection_lengths), injection_count)
                    self.assertTrue(all(0 < length <= block for length in base_lengths))
                    self.assertTrue(
                        all(0 < length <= block for length in injection_lengths)
                    )
                capacity = (population + block - 1) // block + 2 * len(base) + 1
                self.assertLessEqual(tasks, capacity)


if __name__ == "__main__":
    unittest.main()
