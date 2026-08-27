from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UGKPFiniteContactContract(unittest.TestCase):
    @staticmethod
    def _function_body(text: str, signature: str) -> str:
        start = text.index(signature)
        opening = text.index("{", start)
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1:index]
        raise AssertionError(f"unterminated function: {signature}")

    def test_model_and_states_are_declared(self) -> None:
        text = (ROOT / "thermal/GpuFiniteWallContact.H").read_text()
        for token in (
            "particleWallTransientRebound = 2",
            "particleWallTransientDeposit = 3",
            "correctedMaximumSpreadFactor",
            "normalizedEffectiveArea",
            "buildFiniteContactRateTable",
            "interpolateFiniteContactRate",
        ):
            self.assertIn(token, text)

    def test_restart_accepts_all_declared_wall_contact_states(self) -> None:
        text = (ROOT / "gpu/GpuResidentStrict.H").read_text()
        self.assertIn("pStuck[i] > 3", text)
        self.assertNotIn("pStuck[i] > 1", text)
        self.assertIn("pStuck[i] == 1 && pStuckFaceId[i] >= 0", text)
        self.assertIn("pStuck[i] == 2 || pStuck[i] == 3", text)
        self.assertIn("pTheta[i] > pContactDuration[i]", text)
        self.assertIn("pContactPeakFraction[i]", text)

    def test_spec_locks_joules_and_time_pixel_transition(self) -> None:
        heat = (ROOT / "thermal/GpuParticleWallContactHeat.H").read_text()
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        self.assertIn("double wallEnergyJ;", heat)
        self.assertIn("const double activeDt = clampRange(dt, 0.0, duration - age0);", cuda)
        self.assertIn("wallStateAtStepStart", cuda)
        self.assertIn("s.pTheta[i] = age1;", cuda)

    def test_case_uses_independent_heat_transfer_efficiencies(self) -> None:
        text = (
            ROOT.parents[1]
            / "examples/thermal/bentSRM_coldWall/constant/particleProperties"
        ).read_text()
        for name in (
            "depositionHeatTransferEfficiency",
            "reflectionHeatTransferEfficiency",
        ):
            match = re.search(rf"{name}\s+([-+0-9.eE]+)\s*;", text)
            self.assertIsNotNone(match)
            value = float(match.group(1))
            self.assertGreater(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertNotIn("thermalContactAreaFraction", text)
        self.assertNotIn("maximumSpreadAreaRatio", text)

    def test_heat_transfer_efficiencies_are_channel_specific_and_heat_only(self) -> None:
        finite = (ROOT / "thermal/GpuFiniteWallContact.H").read_text()
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        relax = self._function_body(cuda, "relaxOneParticleToResidentGas")
        self.assertNotIn("normalizedLockedKinematicArea", finite)
        self.assertNotIn("maximumSpreadAreaRatio", finite)
        self.assertIn("particleWallDepositionHeatTransferEfficiency", cuda)
        self.assertIn("particleWallReflectionHeatTransferEfficiency", cuda)
        self.assertIn("contactAreaMid", relax)
        self.assertIn("finiteContactWallConductanceTimeIntegral", relax)
        self.assertIn("lumpedParticleWallInterfaceContact", relax)
        self.assertNotIn("localMaximumArea", relax)
        self.assertNotIn("min(equilibriumArea, localMaximumArea)", relax)
        interface = (ROOT / "thermal/GpuWallInterfaceHeat.H").read_text()
        self.assertIn("wallInterfaceConductanceTimeIntegral", interface)
        self.assertIn("particleAdditionalResistanceM2KW", interface)
        self.assertNotIn("thermalContactAreaFraction", cuda)

    def test_rebound_detaches_when_effective_area_is_exhausted(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        body = self._function_body(cuda, "relaxOneParticleToResidentGas")
        rebound = body[
            body.index("particleWallTransientRebound", body.index("bool detach")):
            body.index("else", body.index("particleWallTransientRebound", body.index("bool detach")))
        ]
        self.assertIn("effectiveContactArea", rebound)
        self.assertRegex(rebound, r"detach\s*=\s*!\(effectiveContactArea\s*>\s*0\.0\)")
        self.assertNotIn("age1 >= duration", rebound)

    def test_contact_duration_uses_asymmetric_inertial_capillary_model(self) -> None:
        finite = (ROOT / "thermal/GpuFiniteWallContact.H").read_text()
        body = self._function_body(finite, "evaluateFiniteWallContactImpact")
        for required in (
            "0.5*beta*diameterM/normalSpeedMps",
            "naturalFrequency = 8.0/capillaryTime",
            "20.0*material.viscosityPaS",
            "peakTimeFraction = spreadingDuration/duration",
        ):
            self.assertIn(required, body)
        for removed in (
            "speedMagnitudeMps",
            "0.9311",
            "1.4050",
            "0.4764",
            "spreadingTimeScale",
            "recoilTimeScale",
        ):
            self.assertNotIn(removed, body)

    def test_host_device_state_mirror_is_not_scrubbed(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        self.assertNotIn("scrubHostCalculationScalars", text)
        self.assertNotIn("UGKP_DIAG", text)
        self.assertIn("s->gasFluxScheme = gasFluxScheme;", text)

    def test_all_conductive_wall_states_share_face_coverage(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        body = self._function_body(
            text, "__global__ void accumulateParticleWallRepresentedContactAreaKernel"
        )
        self.assertNotIn(
            "s.pStuck[i] != Foam::gpuThermal::particleWallDeposited", body
        )
        for token in (
            "particleWallTransientRebound",
            "particleWallTransientDeposit",
            "pContactDuration[i]",
            "pContactMaximumArea[i]",
            "pContactPeakFraction[i]",
            "normalizedKinematicArea",
            "pDepositionArea[i]",
            "representedDepositionContactArea",
        ):
            self.assertIn(token, body)
        thermal = (ROOT / "thermal/GpuParticleWallContactHeat.H").read_text()
        self.assertNotIn("particleWallMaximumDepositionCoverage", thermal)
        self.assertNotIn("depositionContactAreaScale", thermal)
        self.assertIn("particleWallMaximumCoverage*s.magSf[faceI]", text)

    def test_wall_energy_warp_group_collective_is_executed_by_all_group_lanes(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        helper = self._function_body(
            text, "__device__ void atomicAddParticleWallEnergyByFace"
        )
        self.assertIn("__shfl_sync(group, wallEnergyJ, sourceLane)", helper)
        self.assertLess(helper.index("while (remaining != 0u)"), helper.index("if (lane == leader)"))
        self.assertNotIn("__shfl_sync(active, wallEnergyJ", helper)

    def test_wall_bound_directory_is_reused_by_area_accumulation(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        area = self._function_body(
            text, "__global__ void accumulateParticleWallRepresentedContactAreaKernel"
        )
        gather = self._function_body(text, "__global__ void gatherSelectedParticlesKernel")
        for token in (
            "wallBoundParticleIndex",
            "wallBoundParticleCountDevice",
            "cudaMalloc wall-bound particle index",
            "cudaMalloc wall-bound particle count",
            "reset wall-bound particle directory count",
            "rebuild restart wall-bound particle directory",
        ):
            self.assertIn(token, text)
        self.assertIn("wallBoundDirectoryParticle(s, entry)", area)
        self.assertIn("wallBoundDirectoryCount(s)", area)
        self.assertNotIn("*s.particleCountDevice", area)
        self.assertIn("publishWallBoundParticleIndex(s, dst)", gather)

    def test_relaxation_splits_mobile_scan_from_wall_bound_directory(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        mobile = self._function_body(
            text, "__global__ void relaxMobileParticlesToResidentGasKernelStatic"
        )
        wall = self._function_body(
            text, "__global__ void relaxWallBoundParticlesToResidentGasKernelStatic"
        )
        self.assertIn("*s.particleCountDevice", mobile)
        self.assertIn("particleWallMobile", mobile)
        self.assertIn("wallBoundDirectoryCount(s)", wall)
        self.assertIn("wallBoundDirectoryParticle(s, entry)", wall)
        self.assertIn("!= Foam::gpuThermal::particleWallMobile", wall)
        self.assertNotIn("relaxParticlesToResidentGasKernelStatic", text)

    def test_coverage_scale_is_frozen_and_shared_by_heat_and_contact_mechanics(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        body = self._function_body(text, "__device__ void relaxOneParticleToResidentGas")
        finite_start = body.index("if (finiteContact)")
        finite_heat = body[finite_start:body.index("s.pTheta[i] = age1", finite_start)]
        compact_heat = re.sub(r"\s+", "", finite_heat)
        self.assertIn(
            "s.particleWallReflectionHeatTransferEfficiency*s.particleWallContactAreaScale[faceI]",
            compact_heat,
        )
        collision = self._function_body(
            text, "__device__ void finalizeOneThermalizedStuckParticle"
        )
        self.assertIn("particleWallContactAreaScale[faceI]", collision)
        self.assertIn("contactAreaScale*adhesionSpecificEnergyPerArea", collision)
        compact_collision = re.sub(r"\s+", "", collision)
        self.assertIn("sampledSpecificEnergy/(contactAreaScale*adhesionSpecificEnergyPerArea)", compact_collision)
        self.assertNotIn("particleWallDepositionHeatTransferEfficiency", collision)
        self.assertNotIn("particleWallReflectionHeatTransferEfficiency", collision)

    def test_face_scale_is_prepared_before_collision_and_reused_by_heat(self) -> None:
        text = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        advance = self._function_body(
            text, 'extern "C" int ugkwpGpuResidentStrictAdvance'
        )
        self.assertEqual(advance.count("prepareParticleWallContactAreaScale"), 1)
        self.assertLess(
            advance.index("prepareParticleWallContactAreaScale"),
            advance.index("samplePoissonPoolParticlesKernel"),
        )
        self.assertLess(
            advance.index("prepareParticleWallContactAreaScale"),
            advance.index("relaxWallBoundParticlesToResidentGasKernelStatic"),
        )

    def test_all_contact_geometry_uses_runtime_constant_angle(self) -> None:
        capillary = (ROOT / "thermal/GpuCapillaryDetachment.H").read_text()
        finite = (ROOT / "thermal/GpuFiniteWallContact.H").read_text()
        self.assertIn("const double contactAngleCosine", capillary)
        self.assertNotIn("referenceContactAngleDegree", capillary)
        self.assertNotIn("50.90", capillary)
        self.assertIn("const double contactAngleCosine", finite)
        self.assertIn("oneMinusCosContactAngle", finite)
        self.assertNotIn("finiteContactAngleRad", finite)
        spread = self._function_body(finite, "correctedMaximumSpreadFactor")
        impact = self._function_body(finite, "evaluateFiniteWallContactImpact")
        self.assertNotIn("::cos", spread)
        self.assertNotIn("::cos", impact)

    def test_small_angle_spread_root_expands_upper_bracket(self) -> None:
        finite = (ROOT / "thermal/GpuFiniteWallContact.H").read_text()
        body = self._function_body(finite, "correctedMaximumSpreadFactor")
        self.assertIn("while (!(upperResidual > 0.0) && expansion < 40)", body)
        self.assertIn("lower = upper;", body)
        self.assertIn("upper *= 2.0;", body)
        self.assertIn("if (!(upperResidual > 0.0))", body)

    def test_cold_wall_deposit_transition_is_recoil_only(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        body = self._function_body(cuda, "__device__ void relaxOneParticleToResidentGas")
        self.assertIn("targetContactArea = fmin(equilibriumArea, maximumArea)", body)
        self.assertNotIn("targetRemainingArea", body)
        self.assertIn("theta1 >= peakTimeFraction", body)
        self.assertNotIn("theta1 >= 0.5", body)
        self.assertIn("fmax(targetContactArea, frozenArea)", body)
        self.assertIn("if (!coldWallContact)", body)
        self.assertIn('asm("trap;")', body)
        self.assertNotIn("particleWallContactAreaScale", body[body.index("bool detach"):body.index("if (detach)")])

    def test_collision_and_recoil_release_velocities_are_exclusive(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        collision = self._function_body(
            cuda, "__device__ void finalizeOneThermalizedStuckParticle"
        )
        relax = self._function_body(cuda, "__device__ void relaxOneParticleToResidentGas")
        self.assertIn("poolMeanUx + fluctuationScale*fluctuationUx", collision)
        collision_detach = collision[
            collision.index("if (damage.detached)"):
            collision.index("else", collision.index("if (damage.detached)"))
        ]
        self.assertIn("s.puxOld[i] = releaseUx", collision_detach)
        self.assertNotIn("s.pux[i] = s.puxOld[i]", collision_detach)
        detach = relax[relax.index("if (detach)"):relax.index("else if (enterLongDeposit)")]
        self.assertIn("s.pux[i] = s.puxOld[i]", detach)
        self.assertNotIn("poolMeanUx", detach)

    def test_face_scale_is_mechanical_even_when_heat_is_disabled(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        prepare = self._function_body(cuda, "int prepareParticleWallContactAreaScale")
        self.assertIn("particleStuckModelConfigured", prepare)
        self.assertNotIn("particleWallHeatTransferEnabled", prepare)

    def test_finite_contact_state_machine_runs_when_heat_is_disabled(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        relax = self._function_body(cuda, "__device__ void relaxOneParticleToResidentGas")
        self.assertIn("if (finiteContact)", relax)
        self.assertNotIn(
            "if (finiteContact && s.particleWallHeatTransferEnabled != 0)",
            relax,
        )
        self.assertRegex(
            relax,
            r"activeDt\s*>\s*0\.0\s*&&\s*"
            r"s\.particleWallHeatTransferEnabled\s*!=\s*0",
        )

    def test_finite_contact_spread_solver_accepts_captured_small_particle(self) -> None:
        source = ROOT / "tests/finite_contact_algebra_test.cpp"
        with tempfile.TemporaryDirectory(prefix="fsh-contact-") as temporary:
            executable = Path(temporary) / "finite_contact_algebra_test"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-I",
                    str(ROOT / "thermal"),
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
