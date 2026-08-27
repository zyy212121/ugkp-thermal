#!/usr/bin/env python3
                                                                         

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="strict")


class UGKPStuckConfigurationContract(unittest.TestCase):
    def test_default_configuration_is_exposed(self) -> None:
        frontend = read("gpu/GpuResidentStrict.H")
        for token in (
            'scalar(20)',
            '"heatTransfer", false',
            'scalar(0.63)',
            '"depositionHeatTransferEfficiency", scalar(1)',
            '"reflectionHeatTransferEfficiency", scalar(1)',
            '"adhesionEnergyScale", scalar(1)',
            '"contactAngleDegree", scalar(145)',
            '"particleStuckWallHeatFlux"',
            '"particleReflectedWallHeatFlux"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, frontend)
        for removed in (
            '"muLiquid"',
            '"sigmaLiquidGas"',
            '"kParticle"',
            '"mechanicalDampingScale"',
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, frontend)

    def test_contact_angle_is_host_validated_before_backend_upload(self) -> None:
        frontend = read("gpu/GpuResidentStrict.H")
        self.assertIn("FatalIOErrorInFunction(ugkwpProps)", frontend)
        self.assertIn("contactAngleDegree <= scalar(0)", frontend)
        self.assertIn("contactAngleDegree >= scalar(180)", frontend)
        self.assertIn("exit(FatalIOError)", frontend)

    def test_all_particle_models_share_rho_s(self) -> None:
        fields = read("createFields.H")
        frontend = read("gpu/GpuResidentStrict.H")
        self.assertIn('ugkwpProps.lookupOrDefault<scalar>("rhoS", 2500.0)', fields)
        self.assertNotIn('"particleCp"', fields)
        self.assertNotIn('"particleRho"', fields)
        self.assertNotIn("const scalar rhoLiquid = rhoSolid;", frontend)

    def test_sommerfeld_is_restricted_to_kind_five(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        kind_five = cuda.index("else if (kind == 5)")
        evaluation = cuda.index("evaluateSommerfeldImpact", kind_five)
        next_branch = cuda.index("\n        else\n", kind_five)
        self.assertLess(evaluation, next_branch)
        self.assertEqual(cuda.count("evaluateSommerfeldImpact"), 1)

    def test_low_k_candidate_uses_patch_restitution(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        kind_five = cuda.index("else if (kind == 5)")
        body = cuda[kind_five : cuda.index("\n        else\n", kind_five)]
        self.assertIn("impact.deposit", body)
        self.assertIn("particleWallTransientRebound", body)
        self.assertIn("particleWallTransientDeposit", body)
        self.assertIn("s.cellFaceRestitution[plane]", body)
        self.assertIn("(1.0 + restitution)*un", body)

    def test_heat_is_write_time_only_on_host(self) -> None:
        solver = read("diluteUgkwpFoam.C")
        self.assertIn("particleWallEnergyAccumulationTime +=", solver)
        self.assertIn("if (runTime.writeTime())", solver)
        self.assertEqual(
            solver.count("downloadAndResetParticleWallHeatLedgers"),
            1,
        )
        self.assertIn("depositedWallEnergyJ", solver)
        self.assertIn("reflectedWallEnergyJ", solver)
        self.assertNotIn("particleWallHeatRate", solver)
        self.assertNotIn("powerW=", solver)

    def test_heat_flux_fields_are_split_by_post_impact_outcome(self) -> None:
        fields = read("createFields.H")
        solver = read("diluteUgkwpFoam.C")
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn("particleStuckWallHeatFluxPtr", fields)
        self.assertIn("particleReflectedWallHeatFluxPtr", fields)
        self.assertIn("particleStuckWallHeatFluxPtr()", solver)
        self.assertIn("particleReflectedWallHeatFluxPtr()", solver)
        self.assertIn("particleWallDepositedEnergy", cuda)
        self.assertIn("particleWallReflectedEnergy", cuda)
        relax_start = cuda.index("__device__ void relaxOneParticleToResidentGas")
        relax_end = cuda.index(
            "__global__ void relaxMobileParticlesToResidentGasKernelStatic",
            relax_start,
        )
        relax_body = cuda[relax_start:relax_end]
        self.assertNotIn("s.particleWallDepositedEnergy", relax_body)
        self.assertIn("s.particleWallReflectedEnergy", relax_body)
        cold_1d = read("../../common/wall/GpuColdWall1DDevice.cuh")
        cold_2d = read("../../common/wall/GpuColdWall2DDevice.cuh")
        self.assertIn("particleWallDepositedEnergy", cold_1d)
        self.assertIn("particleWallDepositedEnergy", cold_2d)
        self.assertIn(
            "faceChannelKey = 2*globalFaceId + ledgerChannel",
            cuda,
        )
        self.assertIn("__match_any_sync(active, faceChannelKey)", cuda)

    def test_heat_transfer_efficiencies_do_not_change_contact_geometry(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        thermal = read("thermal/GpuParticleWallContactHeat.H")
        self.assertNotIn("particleWallMaximumSpreadAreaRatio", cuda)
        self.assertIn("particleWallDepositionHeatTransferEfficiency", cuda)
        self.assertIn("particleWallReflectionHeatTransferEfficiency", cuda)
        cold_1d = read("../../common/wall/GpuColdWall1DDevice.cuh")
        cold_2d = read("../../common/wall/GpuColdWall2DDevice.cuh")
        finite = read("thermal/GpuFiniteWallContact.H")
        self.assertNotIn("normalizedLockedKinematicArea", finite)
        self.assertIn(
            "(finiteContactPi/4.0)*beta*beta*diameterM*diameterM",
            finite,
        )
        self.assertNotIn("localMaximumArea", cuda)
        self.assertIn("capillary.equilibriumContactAreaM2", cuda)
        self.assertNotIn("depositionHeatTransferEfficiency", thermal)
        for cold in (cold_1d, cold_2d):
            self.assertIn("particleWallDepositionHeatTransferEfficiency", cold)
            self.assertIn("particleWallReflectionHeatTransferEfficiency", cold)
        self.assertNotIn("thermalContactAreaM2", thermal)
        self.assertNotIn("thermalContactAreaFraction", cuda)

    def test_stuck_particles_exchange_heat_with_gas_and_wall(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        relax_start = cuda.index(
            "__device__ void relaxOneParticleToResidentGas"
        )
        relax_end = cuda.index(
            "__global__ void relaxMobileParticlesToResidentGasKernelStatic",
            relax_start,
        )
        relax_body = cuda[relax_start:relax_end]
        self.assertIn("s.solveParticleTemperature != 0", relax_body)
        self.assertNotIn("&& !stuck", relax_body)
        self.assertIn("&& !coldWallContact", relax_body)
        self.assertIn("if (finiteContact)", relax_body)
        self.assertRegex(
            relax_body,
            r"activeDt\s*>\s*0\.0\s*&&\s*"
            r"s\.particleWallHeatTransferEnabled\s*!=\s*0",
        )
        self.assertIn("wallStateAtStepStart == Foam::gpuThermal::particleWallDeposited", relax_body)
        self.assertIn("relaxColdWall1DParticlesToResidentGasKernel", cuda)
        cold_wall = read("../../common/wall/GpuColdWall1DDevice.cuh")
        self.assertIn("advanceColdWall1DThermalGroup", cold_wall)
        self.assertIn("particleWallSolidifyingDeposition", cold_wall)

        heat_start = cuda.index(
            "__global__ void applyEulerianParticleMaterialHeatKernel"
        )
        heat_end = cuda.index(
            "__device__ unsigned long long mixSeed",
            heat_start,
        )
        heat_body = cuda[heat_start:heat_end]
        self.assertIn("finiteOr(s.momRhoP[c]", heat_body)
        self.assertNotIn("stuckThermal", cuda)

        self.assertNotIn("mobileThermalRhoP", cuda)
        self.assertEqual(cuda.count("blockReduceComponentSums<11>"), 0)

    def test_component_reduction_uses_a_converged_full_warp(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        start = cuda.index("template<int NumComponents>")
        end = cuda.index(
            "__global__ void accumulatePoissonPoolParticlesByCellKernel",
            start,
        )
        body = cuda[start:end]
        self.assertIn("constexpr unsigned int fullWarpMask = 0xffffffffu;", body)
        self.assertIn("__syncwarp(fullWarpMask);", body)
        self.assertNotIn("= __activemask();", body)
        self.assertNotIn("__popc(warpMask)", body)

    def test_stuck_collision_sampling_uses_energy_barrier(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        sample_start = cuda.index("__device__ void sampleOnePoissonPoolParticle")
        sample_end = cuda.index(
            "__device__ void finalizeOneThermalizedMobileParticle",
            sample_start,
        )
        sample_body = cuda[sample_start:sample_end]
        self.assertIn("const bool wasStuck = s.pStuck[i] != 0;", sample_body)
        self.assertNotIn("s.pT[i] =", sample_body)
        self.assertNotIn("s.pStuck[i] = 0;", sample_body)

        finalize_start = cuda.index(
            "__device__ void finalizeOneThermalizedStuckParticle"
        )
        finalize_end = cuda.index("template<bool StuckPath>", finalize_start)
        finalize_body = cuda[finalize_start:finalize_end]
        self.assertIn("evaluateCapillaryDetachmentState", finalize_body)
        self.assertIn("applyCapillaryContactDamage", finalize_body)
        self.assertIn("adhesionSpecificEnergyPerArea", finalize_body)
        self.assertIn("accumulatedDamageArea", finalize_body)
        self.assertIn("damage.remainingContactAreaM2", finalize_body)
        self.assertNotIn("accumulatedSpecificEnergy", finalize_body)
        self.assertIn("s.pStuck[i] = 0;", finalize_body)
        self.assertIn("s.pStuckFaceId[i] = -1;", finalize_body)
        self.assertIn("s.pDepositionArea[i] = 0.0f;", finalize_body)

        self.assertNotIn(
            "accumulateUnresolvedThetaPoolParticlesByCellKernel", cuda
        )
        poisson_start = cuda.index(
            "accumulatePoissonPoolParticlesByCellKernel"
        )
        poisson_end = cuda.index("template<bool PoissonMode>", poisson_start)
        self.assertIn("particleMomentThetaDevice(s, i)", cuda[poisson_start:poisson_end])
        self.assertNotIn("poissonPoolHeat", cuda)

    def test_v3_metadata_crosses_all_layers(self) -> None:
        active = "\n".join(
            read(path)
            for path in (
                "gpu/GpuBackendApi.H",
                "gpu/GpuBackendClient.C",
                "private_backend/GpuBackendServer.C",
                "gpu/GpuResidentStrict.H",
                "private_backend/GpuResidentStrict.cu",
            )
        )
        self.assertIn("UGKP_PARTICLES_SCHEMA3", active)
        self.assertGreaterEqual(active.count("pStuckFaceId"), 20)
        self.assertGreaterEqual(active.count("pDepositionArea"), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
