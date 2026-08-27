from __future__ import annotations

import re
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BRANCHES = {
    "gasUGKP": (
        ROOT / "applications/gasUGKP/readGpuGasConfiguration.H",
        ROOT / "applications/gasUGKP/gpu/GpuResidentStrict.H",
        ROOT / "applications/gasUGKP/private_backend/GpuResidentStrict.cu",
    ),
    "FSH": (
        ROOT / "applications/FSH/readGpuGasConfiguration.H",
        ROOT / "applications/FSH/gpu/GpuResidentStrict.H",
        ROOT / "applications/FSH/private_backend/GpuResidentStrict.cu",
    ),
    "CHT": (
        ROOT / "applications/CHT/readGpuGasConfiguration.H",
        ROOT / "applications/CHT/gpu/GpuResidentStrict.H",
        ROOT / "applications/CHT/gpu/GpuResidentStrict.cu",
    ),
}


class UnifiedSchedulingToolsContract(unittest.TestCase):
    def test_shared_configuration_exists_and_reads_scheduling_dictionary(self) -> None:
        shared = ROOT / "common/GpuSchedulingConfiguration.H"
        self.assertTrue(shared.is_file())
        text = shared.read_text(encoding="utf-8")
        for token in (
            "gpuResidentPureGasOnly",
            "gpuCsrLevel",
            "gpuCsrHeavyReductionAutoInterval",
            "gpuParticleBlockThreads",
            "gpuReductionBlockThreads",
        ):
            self.assertIn(token, text)
        self.assertRegex(text, r"particleBlockThreads\s*=\s*128")
        self.assertRegex(text, r"reductionBlockThreads\s*=\s*128")
        self.assertNotIn('"gpuFixedWorkBlockThreads"', text)
        for name, (reader, _, _) in BRANCHES.items():
            source = reader.read_text(encoding="utf-8")
            self.assertIn('"schedulingProperties"', source, name)
            self.assertIn("GpuSchedulingConfiguration", source, name)

    def test_three_backends_store_distinct_b1_b2_b3(self) -> None:
        for name, (_, _, cuda_path) in BRANCHES.items():
            cuda = cuda_path.read_text(encoding="utf-8")
            self.assertIn("fixedCellBlockThreads", cuda, name)
            self.assertIn("fixedFaceBlockThreads", cuda, name)
            self.assertIn("fixedWorkBlockTuned", cuda, name)
            self.assertIn("particleBlockThreads", cuda, name)
            self.assertIn("reductionBlockThreads", cuda, name)
            self.assertRegex(cuda, r"tuneFixedWorkBlockThreads\s*\(", name)

    def test_tool_b1_uses_deferred_cuda_event_measurement(self) -> None:
        shared_tool = (ROOT / "common/GpuToolB1.cuh").read_text(encoding="utf-8")
        candidate_pattern = re.compile(
            r"\{\s*32\s*,\s*64\s*,\s*96\s*,\s*128\s*,\s*160\s*,"
            r"\s*192\s*,\s*224\s*,\s*256\s*\}"
        )
        for name, (_, _, cuda_path) in BRANCHES.items():
            cuda = cuda_path.read_text(encoding="utf-8")
            implementation = cuda if name == "gasUGKP" else cuda + shared_tool
            self.assertRegex(implementation, candidate_pattern, name)
            self.assertRegex(implementation, r"toolB1WarmupRuns\s*=\s*1", name)
            self.assertRegex(implementation, r"toolB1MeasuredRuns\s*=\s*5", name)
            for token in (
                "cudaEventCreate",
                "cudaEventRecord",
                "cudaEventElapsedTime",
                "std::nth_element",
            ):
                self.assertIn(token, implementation, name)
            self.assertNotIn("2*faceWaves + cellWaves", cuda, name)
            self.assertNotIn("2*((faceGrid + faceCapacity", cuda, name)
            tool_region = re.search(
                r"int launchToolB1CellBundle[\s\S]*?"
                r"int tuneFixedWorkBlockThreads[\s\S]*?\n\}",
                implementation,
            )
            self.assertIsNotNone(tool_region, name)
            for forbidden in (
                "applyGasFluxDivergenceByCellKernel",
                "blendGasConservativeStateKernel",
                "applyGasGravityKernel",
                "applyEulerianGasSolidCouplingKernel",
            ):
                self.assertNotIn(forbidden, tool_region.group(0), name)

            configure = re.search(
                r"int configure(?:ParticleLaunchGeometry|LaunchOccupancy)"
                r"\s*\([^)]*\)\s*\{"
                r"(?P<body>[\s\S]*?)\n\}",
                cuda,
            )
            self.assertIsNotNone(configure, name)
            self.assertNotIn("tuneFixedWorkBlockThreads", configure.group("body"), name)

            advance = re.search(
                r'extern "C" int ugkwpGpuResidentStrictAdvance(?:GasOnly)?'
                r"\s*\([^)]*\)\s*\{(?P<body>[\s\S]*?)\n\}",
                cuda,
            )
            self.assertIsNotNone(advance, name)
            self.assertIn("tuneFixedWorkBlockThreads", advance.group("body"), name)

    def test_l2_auto_has_interval_and_tool_b3(self) -> None:
        shared = (ROOT / "common/GpuSchedulingConfiguration.H").read_text(
            encoding="utf-8"
        )
        self.assertIn("gpuCsrHeavyReductionAutoInterval", shared)
        for name, (_, header_path, cuda_path) in BRANCHES.items():
            header = header_path.read_text(encoding="utf-8")
            cuda = cuda_path.read_text(encoding="utf-8")
            self.assertIn("GpuSchedulingConfiguration", header, name)
            self.assertIn("runToolB3", cuda, name)
            self.assertIn("csrHeavyReductionMode", cuda, name)
            self.assertIn("csrHeavyAutoInterval", cuda, name)
            self.assertIn("csrHeavyReductionActive", cuda, name)

    def test_periodic_flux_corrections_are_guarded_by_mesh_flag(self) -> None:
        for name, (_, _, cuda_path) in BRANCHES.items():
            cuda = cuda_path.read_text(encoding="utf-8")
            self.assertIn("hasPeriodicFaces", cuda, name)
            self.assertRegex(
                cuda,
                r"if\s*\(\s*s->hasPeriodicFaces\s*!=\s*0\s*\)\s*\{[\s\S]*?"
                r"enforcePeriodicGasFluxAntisymmetryKernel",
                name,
            )

    def test_examples_own_scheduling_not_physics_keys(self) -> None:
        cases = [
            path.parent.parent
            for path in (ROOT / "examples").glob("**/constant/particleProperties")
        ]
        self.assertTrue(cases)
        moved = (
            "gpuResidentStrict",
            "gpuResidentPureGasOnly",
            "gpuResidentDynamicInlet",
            "gpuResidentParticleCapacity",
            "gpuResidentMaxFaceWalkHops",
            "gpuResidentCourantUpdateInterval",
            "gpuResidentMaxDeltaTGrowth",
            "gpuCsrCellLocalPath",
            "gpuCsrHeavyReduction",
            "gpuCsrWarpAggregatedBinning",
        )
        for case in cases:
            scheduling = case / "constant/schedulingProperties"
            self.assertTrue(scheduling.is_file(), str(case))
            schedule_text = scheduling.read_text(encoding="utf-8")
            particle_text = (case / "constant/particleProperties").read_text(
                encoding="utf-8"
            )
            for key in moved:
                self.assertNotRegex(particle_text, rf"\b{key}\b", str(case))
            self.assertRegex(
                schedule_text,
                r"(?m)^\s*gpuCsrLevel\s+(?:L0|L1|L2|auto)\s*;",
                str(case),
            )

    def test_migration_maps_old_l2_and_is_idempotent(self) -> None:
        migration_path = ROOT / "tools/migrate_scheduling_properties.py"
        spec = importlib.util.spec_from_file_location(
            "migrate_scheduling_properties", migration_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            constant = Path(tmp) / "constant"
            constant.mkdir()
            particle = constant / "particleProperties"
            particle.write_text(
                "parcelMass 5e-9;\n"
                "gpuResidentStrict true;\n"
                "gpuCsrCellLocalPath true;\n"
                "gpuCsrHeavyReduction true;\n"
                "gpuCsrWarpAggregatedBinning true;\n"
                "gpuCsrSplitPreDirectory true;\n",
                encoding="utf-8",
            )
            module.migrate(particle)
            module.migrate(particle)
            schedule_text = (constant / "schedulingProperties").read_text(
                encoding="utf-8"
            )
            particle_text = particle.read_text(encoding="utf-8")
            self.assertRegex(schedule_text, r"(?m)^gpuCsrLevel\s+L2;")
            for key in (
                "gpuResidentStrict",
                "gpuCsrCellLocalPath",
                "gpuCsrHeavyReduction",
                "gpuCsrWarpAggregatedBinning",
                "gpuCsrSplitPreDirectory",
            ):
                self.assertNotRegex(schedule_text, rf"(?m)^\s*{key}\b")
                self.assertNotRegex(particle_text, rf"(?m)^\s*{key}\b")
            self.assertIn("parcelMass 5e-9;", particle_text)


if __name__ == "__main__":
    unittest.main()
