#!/usr/bin/env python3
                                                      

from __future__ import annotations

import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"required UGKP file is missing: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


def alpha_bar(dt_over_tau: float) -> float:
    if dt_over_tau == 0.0:
        return 1.0
    if abs(dt_over_tau) < 1.0e-7:
        x = dt_over_tau
        return 1.0 - 0.5*x + x*x/6.0 - x*x*x/24.0
    return -math.expm1(-dt_over_tau)/dt_over_tau


def csr_task_bound(capacity: int, threshold: int, tile: int) -> int:
    return (capacity + tile - 1)//tile + capacity//(threshold + 1) + 1


class NumericalContracts(unittest.TestCase):
    def test_time_averaged_bgk_weight(self) -> None:
        self.assertEqual(alpha_bar(0.0), 1.0)
        self.assertAlmostEqual(alpha_bar(1.0), 1.0 - math.exp(-1.0), places=15)
        self.assertAlmostEqual(alpha_bar(1.0e8), 1.0e-8, places=20)

    def test_heavy_task_capacity_bound(self) -> None:
        threshold, tile = 8, 6
        for counts in ((9, 0, 0), (17, 5, 42), (100, 100, 100)):
            actual = sum((n + tile - 1)//tile for n in counts if n > threshold)
            self.assertLessEqual(actual, csr_task_bound(sum(counts), threshold, tile))


class SourceContracts(unittest.TestCase):
    def test_independent_version_name(self) -> None:
        make_files = read("Make/files")
        build = read("tools/build_cuda_solver.sh")
        self.assertIn("EXE = $(FOAM_USER_APPBIN)/CHT", make_files)
        self.assertNotIn("diluteUgkwpFoam_UGKP_CHT", make_files)
        self.assertIn(
            'lib_path="${solver_root}/gpu/libugkpcht_cuda_internal.a"',
            build,
        )
        self.assertIn(
            'cuda_bin="${FOAM_USER_APPBIN}/CHT"',
            build,
        )
        self.assertIn('UGKWP_CUDA_EXE_LIBS="${lib_path}', build)

    def test_gks_les_controls_and_gpu_path(self) -> None:
        fields = read("createFields.H") + read("readGpuGasConfiguration.H")
        header = read("gpu/GpuResidentStrict.H")
        cuda = read("gpu/GpuResidentStrict.cu")
        for token in (
            "fluxScheme", "SLAU2.2", "turbulence", "laminar",
            "WALE", "Smagorinsky", "lesDeltaCoeff", "turbulentPrandtl",
            "maxDiffusionNumber",
        ):
            self.assertIn(token, fields)
        for token in (
            "computeGasPrimitiveGradients", "computeGasGradientLimiter",
            "computeGasEddyViscosity", "computeRiemannGasFaceFluxDevice",
            "advanceGasFluxStage", "downloadNutToHostMirror",
        ):
            self.assertIn(token, cuda + header)
        self.assertIn("energyFlux -= kEffective*normalTemperatureGradient;", cuda)

    def test_csr_hierarchy_and_warp_binning(self) -> None:
        header = read("gpu/GpuResidentStrict.H")
        cuda = read("gpu/GpuResidentStrict.cu")
        scheduling = read("../../common/GpuSchedulingConfiguration.H")
        for token in (
            "gpuCsrHeavyReduction", "gpuCsrWarpAggregatedBinning",
            "gpuParticleBlockThreads", "gpuReductionBlockThreads",
        ):
            self.assertIn(token, scheduling)
        for token in (
            "prepareCsrSegmentedReductionTasks",
            "accumulateCsrSegmentedPoolTasksPersistentKernel",
            "finalizeCsrHeavyPoolCellsKernel",
            "accumulateCsrSegmentedMomentTasksPersistentKernel",
            "finalizeCsrHeavyMomentCellsKernel", "__match_any_sync",
        ):
            self.assertIn(token, cuda)

    def test_cht_impact_heat_and_mobile_packing_projection_are_preserved(self) -> None:
        main = read("diluteUgkwpFoam.C")
        header = read("gpu/GpuResidentStrict.H")
        cuda = read("gpu/GpuResidentStrict.cu")
        thermal = read("thermal/GpuThermalExchangeState.C") + read(
            "thermal/GpuParticleWallContactHeat.H"
        )
        for token in (
            "GpuSolidThermalCoupler", "ThermalExchangeRestartState",
            "applyParticleRadiationEnergy", "particleWallDepositedEnergy",
            "particleWallReflectedEnergy",
            "writeSourceResidualRestartMirror",
        ):
            self.assertIn(token, main + header + cuda)
        self.assertIn("UGKP_PARTICLES_SCHEMA4", thermal)
        self.assertIn("UGKP_PARTICLES_SCHEMA1_BIN", thermal)
        self.assertNotIn("depositedParticleWallContact", thermal)
        self.assertIn("evaluateFiniteWallContactImpact", cuda)
        self.assertIn("wallBoundDirectoryParticle", cuda)
        self.assertNotIn("particleWallMaximumDepositionCoverage", thermal)
        self.assertNotIn("jammingPressureFromEpsilonDevice", cuda)
        self.assertNotIn("const double pTotal = pColl + pJam;", cuda)
        for token in (
            "accumulateMobilePackingMomentsKernel",
            "prepareMobilePackingProjectionKernel",
            "solveActiveMobilePackingPressureJacobiKernel",
            "computeMobilePackingFaceCorrectionFluxKernel",
            "reconstructMobilePackingVelocityCorrectionKernel",
            "applyMobilePackingCorrectionToParticlesKernel",
            "applyMobilePackingProjection(s, dt, block)",
        ):
            self.assertIn(token, cuda)
        self.assertIn("gpuResidentPackingProjectionIterations", header)
        self.assertIn("thermalExchangeState", thermal)

    def test_cht_wall_ledger_wraps_final_scaled_gks_flux_once(self) -> None:
        cuda = read("gpu/GpuResidentStrict.cu")
        self.assertEqual(
            cuda.count("gasWallEnergy[f] += ledgerDt*s.gasPhiRhoE[f];"), 1
        )
        divergence = cuda.split(
            "__global__ void applyGasFluxDivergenceByCellKernel", 1
        )[1].split("__global__ void recoverPrimitivesKernel", 1)[0]
        self.assertIn("dRhoE += sign*s.gasPhiRhoE[faceI];", divergence)
        self.assertNotIn("gasWallEnergy[faceI]", divergence)

    def test_single_collision_pool_and_packed_cht_rebuild_remain(self) -> None:
        cuda = read("gpu/GpuResidentStrict.cu")
        for token in (
            "accumulatePoissonPoolParticlesByCellKernel",
            "accumulateParticleMomentsSegmentedKernel",
            "launchCsrHeavyPoolReduction",
            "launchCsrHeavyMomentReduction",
        ):
            self.assertIn(token, cuda)
        self.assertNotIn(
            "accumulateUnresolvedThetaPoolParticlesByCellKernel", cuda
        )
        self.assertNotIn(
            "launchCsrHeavyPoolReduction(s, 0.0, false, block)", cuda
        )
        packed_refresh = cuda.split(
            'extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked', 1
        )[1]
        packed_refresh = packed_refresh.split(
            'extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev', 1
        )[0]
        self.assertIn("refreshPackedParticleEnthalpyKernel", packed_refresh)
        self.assertIn("accumulateParticleEnthalpyMomentAtomicKernel", packed_refresh)
        self.assertNotIn("binParticlesByCell", packed_refresh)
        self.assertNotIn("initialiseEpsGPrevKernel", packed_refresh)
        self.assertNotIn("rebuildResidentParticleMomentsFromParticles", packed_refresh)
        rebuild = cuda.split("int rebuildResidentParticleMomentsFromParticles", 1)[1]
        rebuild = rebuild.split("int preparePreTransportParticleDirectory", 1)[0]
        self.assertIn("if (s->csrCellLocalPathEnabled != 0)", rebuild)
        self.assertIn("accumulateParticleMomentsSegmentedKernel", rebuild)

    def test_production_source_has_no_development_probe(self) -> None:
        cuda = read("gpu/GpuResidentStrict.cu")
        build = read("tools/build_cuda_solver.sh") + read("Make/options")
        self.assertIn("#ifdef UGKP_DEVELOPMENT_PROBES", cuda)
        self.assertIn("DevelopmentAdvanceProbe", cuda)
        self.assertNotIn("-DUGKP_DEVELOPMENT_PROBES", build)

    def test_no_cpu_turbulence_discretisation(self) -> None:
        combined = "\n".join(
            read(path) for path in (
                "createFields.H", "diluteUgkwpFoam.C",
                "gpu/GpuResidentStrict.H", "gpu/GpuResidentStrict.cu",
            )
        )
        for pattern in (
            r"turbulenceModel\s*::\s*New", r"LESModel\s*::\s*New",
            r"RASModel\s*::\s*New", r"fvc\s*::\s*grad",
            r"fvm\s*::\s*laplacian",
        ):
            self.assertIsNone(re.search(pattern, combined))


if __name__ == "__main__":
    unittest.main(verbosity=2)
