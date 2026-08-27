#!/usr/bin/env python3
                                                           

                                                                            
                                                                             
              
   

from __future__ import annotations

import math
import re
import unittest
from itertools import product
from pathlib import Path

try:
    from .source_contract_utils import code_only, function_block
except ImportError:
    from source_contract_utils import code_only, function_block


ROOT = Path(__file__).resolve().parents[1]


def positive_half_first_moment(mean: float, variance: float) -> float:
                                                     
    sigma = math.sqrt(variance)
    z = mean / sigma
    probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    thermal = sigma * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return mean * probability + thermal


def negative_half_first_moment(mean: float, variance: float) -> float:
                                                     
    sigma = math.sqrt(variance)
    z = mean / sigma
    probability = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    thermal = sigma * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return mean * probability - thermal


def alpha_bar(dt_over_tau: float) -> float:
                                                                    
    if dt_over_tau == 0.0:
        return 1.0
    if abs(dt_over_tau) < 1.0e-7:
        x = dt_over_tau
        return 1.0 - 0.5 * x + x * x / 6.0 - x * x * x / 24.0
    return -math.expm1(-dt_over_tau) / dt_over_tau


def trace_free_strain(gradient: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    divergence = sum(gradient[i][i] for i in range(3))
    return [
        [
            0.5 * (gradient[i][j] + gradient[j][i])
            - (divergence / 3.0 if i == j else 0.0)
            for j in range(3)
        ]
        for i in range(3)
    ]


def newtonian_stress(
    gradient: tuple[tuple[float, ...], ...], dynamic_viscosity: float
) -> list[list[float]]:
    strain = trace_free_strain(gradient)
    return [
        [2.0 * dynamic_viscosity * strain[i][j] for j in range(3)]
        for i in range(3)
    ]


def smagorinsky_nut(
    gradient: tuple[tuple[float, ...], ...], cs: float, delta: float
) -> float:
    strain = trace_free_strain(gradient)
    strain_sq = sum(strain[i][j] ** 2 for i in range(3) for j in range(3))
    return (cs * delta) ** 2 * math.sqrt(2.0 * strain_sq)


def wale_nut(
    gradient: tuple[tuple[float, ...], ...], cw: float, delta: float
) -> float:
    strain = trace_free_strain(gradient)
    strain_sq = sum(strain[i][j] ** 2 for i in range(3) for j in range(3))
    gradient_sq = [
        [sum(gradient[i][k] * gradient[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    trace_gradient_sq = sum(gradient_sq[i][i] for i in range(3))
    sd = [
        [
            0.5 * (gradient_sq[i][j] + gradient_sq[j][i])
            - (trace_gradient_sq / 3.0 if i == j else 0.0)
            for j in range(3)
        ]
        for i in range(3)
    ]
    sd_sq = sum(sd[i][j] ** 2 for i in range(3) for j in range(3))
    if sd_sq == 0.0:
        return 0.0
    denominator = strain_sq ** 2.5 + sd_sq ** 1.25
    return (cw * delta) ** 2 * sd_sq ** 1.5 / denominator


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"required UGKP file is missing: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


def csr_heavy_tasks(
    offsets: list[int], threshold: int, tile_particles: int
) -> dict[int, list[tuple[int, int]]]:
    tasks: dict[int, list[tuple[int, int]]] = {}
    for cell, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if end - begin <= threshold:
            continue
        tasks[cell] = [
            (tile_begin, min(tile_begin + tile_particles, end))
            for tile_begin in range(begin, end, tile_particles)
        ]
    return tasks


class NumericalContractTests(unittest.TestCase):
    def test_half_maxwell_normalisation_at_zero_mean(self) -> None:
        variance = 7.5
        expected = math.sqrt(variance / (2.0 * math.pi))
        self.assertAlmostEqual(positive_half_first_moment(0.0, variance), expected, places=14)
        self.assertAlmostEqual(negative_half_first_moment(0.0, variance), -expected, places=14)

                                                                              
        legacy_prefactor = math.sqrt(variance / math.pi)
        self.assertAlmostEqual(legacy_prefactor / expected, math.sqrt(2.0), places=14)

    def test_half_maxwell_parts_recover_full_first_moment(self) -> None:
        for mean, variance in ((-3.0, 0.4), (-0.2, 2.0), (1.7, 5.0), (20.0, 0.5)):
            recovered = positive_half_first_moment(mean, variance) + negative_half_first_moment(
                mean, variance
            )
            self.assertAlmostEqual(recovered, mean, places=13)

    def test_alpha_bar_limits_and_reference_value(self) -> None:
        self.assertEqual(alpha_bar(0.0), 1.0)
        self.assertAlmostEqual(alpha_bar(1.0e-12), 1.0 - 0.5e-12, places=15)
        self.assertAlmostEqual(alpha_bar(1.0), 1.0 - math.exp(-1.0), places=15)
        self.assertAlmostEqual(alpha_bar(1.0e8), 1.0e-8, places=20)
        self.assertGreater(alpha_bar(1.0e-3), alpha_bar(1.0))
        self.assertGreater(alpha_bar(1.0), alpha_bar(10.0))

    def test_newtonian_stress_reference_tensor(self) -> None:
        gradient = (
            (1.0, 2.0, 3.0),
            (-1.0, 4.0, 5.0),
            (0.5, -2.0, 6.0),
        )
        stress = newtonian_stress(gradient, 2.5)
        expected = (
            (-13.333333333333332, 2.5, 8.75),
            (2.5, 1.6666666666666674, 7.5),
            (8.75, 7.5, 11.666666666666668),
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress[i][j], expected[i][j], places=13)
                self.assertAlmostEqual(stress[i][j], stress[j][i], places=14)
        self.assertAlmostEqual(sum(stress[i][i] for i in range(3)), 0.0, places=13)

    def test_fourier_heat_flux_reference_vector(self) -> None:
        conductivity = 0.7
        grad_t = (2.0, -3.0, 4.0)
        heat_flux = tuple(-conductivity * value for value in grad_t)
        self.assertEqual(heat_flux, (-1.4, 2.0999999999999996, -2.8))

    def test_smagorinsky_reference_value(self) -> None:
        gradient = (
            (1.0, 2.0, 3.0),
            (-1.0, 4.0, 5.0),
            (0.5, -2.0, 6.0),
        )
        self.assertAlmostEqual(
            smagorinsky_nut(gradient, cs=0.17, delta=0.25),
            0.012459634172958627,
            places=15,
        )

    def test_wale_reference_and_zero_gradient(self) -> None:
        gradient = (
            (1.0, 2.0, 3.0),
            (-1.0, 4.0, 5.0),
            (0.5, -2.0, 6.0),
        )
        self.assertAlmostEqual(
            wale_nut(gradient, cw=0.325, delta=0.25),
            0.03159733510026438,
            places=15,
        )
        zero = ((0.0, 0.0, 0.0),) * 3
        self.assertEqual(wale_nut(zero, cw=0.325, delta=0.25), 0.0)

    def test_csr_heavy_tasks_cover_only_heavy_segments_once(self) -> None:
        offsets = [0, 4, 12, 13, 30]
        tasks = csr_heavy_tasks(offsets, threshold=8, tile_particles=6)
        self.assertEqual(tasks, {3: [(13, 19), (19, 25), (25, 30)]})
        covered = [position for begin, end in tasks[3] for position in range(begin, end)]
        self.assertEqual(covered, list(range(13, 30)))

    def test_csr_heavy_task_capacity_bound(self) -> None:
        threshold = 2
        tile_particles = 3
        for counts in product(range(9), repeat=4):
            particle_capacity = sum(counts)
            actual_tasks = sum(
                (count + tile_particles - 1) // tile_particles
                for count in counts
                if count > threshold
            )
            allocated_tasks = (
                (particle_capacity + tile_particles - 1) // tile_particles
                + particle_capacity // (threshold + 1)
                + 1
            )
            self.assertLessEqual(actual_tasks, allocated_tasks)

    def test_warp_group_reservation_produces_unique_dense_ranks(self) -> None:
        cells = [4, 4, 9, 4, 9, 12, 12, 12]
        for cell in sorted(set(cells)):
            lanes = [lane for lane, value in enumerate(cells) if value == cell]
            lane_ranks = {
                lane: sum(other < lane for other in lanes)
                for lane in lanes
            }
            self.assertEqual(sorted(lane_ranks.values()), list(range(len(lanes))))


class SourceContractTests(unittest.TestCase):
    def test_versioned_executable_and_backend_names(self) -> None:
        make_files = read("Make/files")
        private_build = read("private_backend/build_private_backend.sh")
        public_build = read("tools/build_public_frontend.sh")
        protocol = read("gpu/GpuBackendProtocol.H")

        self.assertIn("$(FOAM_USER_APPBIN)/gasUGKP", make_files)
        self.assertIn("gasUGKPCudaBackend", private_build)
        self.assertIn("gasUGKP", private_build)
        self.assertIn("gasUGKP", public_build)
        self.assertIn('frontend="${FOAM_USER_APPBIN}/gasUGKP"', public_build)
        self.assertRegex(protocol, r'0x47554750U\s*;')

    def test_exact_case_configuration_names_are_parsed(self) -> None:
        create_fields = (
            read("readGpuGasConfiguration.H")
            + "\n"
            + read("createFields.H")
        )
        required = (
            "rusanovTadmor",
            "hllKurganov",
            "hlle",
            "hllc",
            "roe",
            "firstOrder",
            "MUSCL",
            "barthJespersen",
            "venkatakrishnan",
            "Euler",
            "SSPRK2",
            "SSPRK3",
            "robustFallback",
            "hllem",
            "hllcAdc",
            "slau2",
            "turbulence",
            "laminar",
            "WALE",
            "Smagorinsky",
            "kOmegaSST",
            "lesDeltaCoeff",
            "turbulentPrandtl",
            "Cw",
            "Cs",
            "maxDiffusionNumber",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, create_fields)

        rejection_contracts = (
            "Unsupported fvSchemes fluxScheme",
            "Unsupported gas reconstruction",
            "Unsupported gas limiter",
            "Unsupported ddtSchemes/default",
            "robustFallback must remain true",
            "Valid values are laminar, WALE, Smagorinsky and kOmegaSST",
            "LES deltaCoeff must be finite and positive",
            "Turbulent Prandtl number must be finite and positive",
            "maxDiffusionNumber must be positive",
        )
        for text in rejection_contracts:
            with self.subTest(rejection=text):
                self.assertIn(text, create_fields)
        self.assertNotIn('"entropyFixCoefficient"', create_fields)
        self.assertNotIn("gasEntropyFixCoefficient", create_fields)

    def test_create_and_courant_ipc_abi_contract(self) -> None:
        protocol = read("gpu/GpuBackendProtocol.H")
        create_match = re.search(r"struct\s+CreateArgs\s*\{(.*?)\};", protocol, re.S)
        self.assertIsNotNone(create_match)
        fields = re.findall(r"(?:std::int32_t|std::uint64_t|double)\s+(\w+)\s*;", create_match.group(1))
        self.assertEqual(
            fields[-31:],
            [
                "collisionalRestitution",
                "pressureKickFraction",
                "jammingPressureEnabled",
                "packingFraction",
                "packingProjectionIterations",
                "gasFluxScheme",
                "gasReconstruction",
                "gasLimiter",
                "gasTimeIntegrator",
                "gasRobustFallback",
                "turbulenceModel",
                "lesDeltaCoeff",
                "turbulentPrandtl",
                "waleCw",
                "smagorinskyCs",
                "maxDiffusionNumber",
                "csrCellLocalPathEnabled",
                "csrHeavyReductionMode",
                "csrHeavyAutoInterval",
                "particleBlockThreads",
                "reductionBlockThreads",
                "csrWarpAggregatedBinning",
                "csrSplitPreDirectoryEnabled",
                "dragModel",
                "dragParameter0",
                "dragParameter1",
                "dragParameter2",
                "dragParameter3",
                "gravityX",
                "gravityY",
                "gravityZ",
            ],
        )
        self.assertIn('protocolMinor = 0', protocol)
        self.assertIn('sizeof(CreateArgs) == 392', protocol)
        self.assertRegex(
            protocol,
            r"struct\s+CourantArgs\s*\{\s*double\s+dt\s*;\s*double\s+targetMaxCo\s*;\s*double\s+scheduleTime\s*;\s*\};",
        )
        for path in (
            "gpu/GpuBackendApi.H",
            "gpu/GpuResidentStrict.H",
            "gpu/GpuBackendClient.C",
            "private_backend/GpuBackendServer.C",
            "private_backend/GpuResidentStrict.cu",
        ):
            with self.subTest(path=path):
                source = read(path)
                self.assertIn("targetMaxCo", source)

    def test_csr_hierarchical_experiment_configuration_contract(self) -> None:
        mirror = read("gpu/GpuResidentStrict.H")
        scheduling = read("../../common/GpuSchedulingConfiguration.H")
        protocol = read("gpu/GpuBackendProtocol.H")
        cuda = read("private_backend/GpuResidentStrict.cu")
        for token in (
            "gpuCsrCellLocalPath",
            "gpuCsrHeavyReduction",
            "gpuParticleBlockThreads",
            "gpuReductionBlockThreads",
            "gpuCsrWarpAggregatedBinning",
        ):
            with self.subTest(token=token):
                self.assertIn(token, scheduling)
        for field in (
            "csrCellLocalPathEnabled",
            "csrHeavyReductionMode",
            "csrHeavyAutoInterval",
            "particleBlockThreads",
            "reductionBlockThreads",
            "csrWarpAggregatedBinning",
        ):
            with self.subTest(field=field):
                self.assertIn(field, protocol)
                self.assertIn(field, cuda)
        self.assertEqual(
            mirror.count("Particle cell path: csrCellLocalPath="),
            1,
        )
        for removed_fixed_control in (
            "gpuCsrHeavyCellThreshold",
            "gpuCsrHeavyTileParticles",
            "gpuCsrHeavyWorkerBlocksPerSM",
        ):
            self.assertNotIn(removed_fixed_control, mirror)

    def test_csr_heavy_and_warp_aggregation_source_contract(self) -> None:
        cuda = "\n".join(
            (
                read("private_backend/GpuResidentStrict.cu"),
                (ROOT.parents[1] / "gpu/CsrSegmentedPoolWorkers.cuh").read_text(),
                (ROOT.parents[1] / "gpu/CsrSegmentedMomentWorkers.cuh").read_text(),
            )
        )
        for token in (
            "prepareCsrSegmentedReductionTasks",
            "countCsrReductionTasksKernel",
            "materializeCsrReductionTasksKernel",
            "csrHeavyCellCount",
            "csrMultiTaskCellList",
            "accumulateCsrSegmentedPoolTasksPersistentKernel",
            "finalizeCsrSegmentedPoolCellsKernel",
            "accumulateCsrSegmentedMomentTasksPersistentKernel",
            "finalizeCsrSegmentedMomentCellsKernel",
            "__match_any_sync",
            "groupMask",
            "laneRank",
            "1 + (count - 1)/s.csrHeavyTileParticles",
        ):
            with self.subTest(token=token):
                self.assertIn(token, cuda)
        self.assertNotRegex(
            cuda,
            r"finalizeCsrSegmented(?:Pool|Moment)CellsKernel\s*<<<\s*s->nCells",
        )
        self.assertNotRegex(
            cuda,
            r"cudaMemcpy(?:Async)?\s*\([^;]*csrHeavyTaskCount",
        )
        self.assertIn("ToolB3 occupancy decision", cuda)

    def test_validation_cases_select_the_expected_models(self) -> None:
        laminar_root = ROOT / "examples/gks_flux_validation"
        les_root = ROOT / "examples/les_validation"
        if not laminar_root.is_dir() or not les_root.is_dir():
            self.skipTest(
                "clean UGKP source package has no legacy validation examples"
            )
        for case in ("sodShockTube", "planarCouette", "fourierSlab"):
            props = read(f"examples/gks_flux_validation/{case}/constant/gksProperties")
            self.assertRegex(props, r"gasNumerics\s*\{")
            self.assertRegex(
                props,
                r"flux\s+(?:rusanovTadmor|hllKurganov|hlle|hllc|roe|hllem)\s*;",
            )
            self.assertNotRegex(props, r"model\s+continuumGKS\s*;")
            self.assertRegex(props, r"model\s+laminar\s*;")
        for case, model in (("waleAffine", "WALE"), ("smagorinskyAffine", "Smagorinsky")):
            props = read(f"examples/les_validation/{case}/constant/gksProperties")
            self.assertRegex(props, rf"model\s+{model}\s*;")
        self.assertTrue((laminar_root / "README.md").is_file())
        self.assertTrue((les_root / "README.md").is_file())

    def test_diffusion_validation_uses_contact_preserving_flux(self) -> None:
        validation_root = ROOT / "examples/gks_flux_validation"
        if not validation_root.is_dir():
            self.skipTest(
                "clean UGKP source package has no legacy validation examples"
            )
        for case in (
            "planarCouette",
            "fourierSlab",
            "fourierSlabPr04",
            "fourierSlabPr08",
        ):
            props = read(
                f"examples/gks_flux_validation/{case}/constant/gksProperties"
            )
            with self.subTest(case=case):
                self.assertRegex(props, r"flux\s+hllc\s*;")

    def test_no_openfoam_cpu_turbulence_or_discretisation_path(self) -> None:
        active_sources = (
            "createFields.H",
            "diluteUgkwpFoam.C",
            "gpu/GpuResidentStrict.H",
            "gpu/GpuBackendClient.C",
            "private_backend/GpuBackendServer.C",
            "private_backend/GpuResidentStrict.cu",
        )
        forbidden = (
            r"turbulenceModel\s*::\s*New",
            r"compressible\s*::\s*turbulenceModel",
            r"LESModel\s*::\s*New",
            r"RASModel\s*::\s*New",
            r"fvc\s*::\s*grad",
            r"fvc\s*::\s*laplacian",
            r"fvm\s*::\s*laplacian",
            r"fvVectorMatrix",
            r"fvScalarMatrix",
        )
        combined = "\n".join(read(path) for path in active_sources)
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, combined))

    def test_cuda_backend_contains_required_gpu_model_markers(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        for token in (
            "computeGasPrimitiveGradients",
            "computeGasGradientLimiter",
            "computeGasEddyViscosity",
            "computeRiemannGasFaceFluxDevice",
            "advanceGasFluxStage",
        ):
            with self.subTest(token=token):
                self.assertIn(token, cuda)

    def test_face_kernel_specialization_preserves_laminar_and_les_paths(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertIn("template<bool IncludeTurbulence>", cuda)
        self.assertIn("if constexpr (IncludeTurbulence)", cuda)
        self.assertIn("s->hostTurbulenceModel == 0", cuda)
        self.assertIn(
            "computeGasInternalFaceFluxKernel<false><<<faceGrid, faceBlock>>>",
            cuda,
        )
        self.assertIn(
            "computeGasInternalFaceFluxKernel<true><<<faceGrid, faceBlock>>>",
            cuda,
        )
        self.assertIn("s->hostTurbulenceModel = turbulenceModel;", cuda)

    def test_continuum_transport_is_separate_from_riemann_flux(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        face_flux = function_block(cuda, "computeRiemannGasFaceFluxDevice")
        active_face_flux = code_only(face_flux)
        self.assertIn("ugkpriemann::fluxUnitArea", active_face_flux)
        self.assertNotRegex(
            active_face_flux,
            r"tauWave|tauJump|numericalDissipationTau|physicalTau|ugkpgks",
        )
        self.assertIn(
            "const double muEffective = s.gasMu + muTurbulent;",
            face_flux,
        )
        self.assertIn(
            "molecularGasConductivity(s) + kTurbulent",
            face_flux,
        )
        self.assertIn("momentumFluxX -= traction.x;", face_flux)
        self.assertIn("momentumFluxY -= traction.y;", face_flux)
        self.assertIn("momentumFluxZ -= traction.z;", face_flux)
        self.assertIn(
            "energyFlux -= kEffective*normalTemperatureGradient;",
            face_flux,
        )
        viscous_block = face_flux[
            face_flux.index("const double muEffective"):
            face_flux.index("massFluxArea = massFlux*area")
        ]
        self.assertNotRegex(
            viscous_block,
            r"gasGradientLimiter(?:Ux|Uy|Uz|T)",
        )
        self.assertRegex(
            face_flux,
            r"if\s*\(\s*muEffective\s*>\s*0\.0\s*\|\|\s*"
            r"kEffective\s*>\s*0\.0\s*\)",
        )
        self.assertIn("ugkpcharacteristic::limitFacePair", active_face_flux)

    def test_muscl_uses_componentwise_bounded_primitive_limiters(self) -> None:
                                                                               
        cuda = read("private_backend/GpuResidentStrict.cu")
        limiter_fields = (
            "gasGradientLimiterRho",
            "gasGradientLimiterUx",
            "gasGradientLimiterUy",
            "gasGradientLimiterUz",
            "gasGradientLimiterP",
            "gasGradientLimiterT",
        )
        for field in limiter_fields:
            with self.subTest(field=field):
                self.assertIn(f"double* {field} = nullptr;", cuda)
                self.assertIn(f"release(s->{field});", cuda)
                self.assertIn(f"allocate(s->{field}, nc", cuda)

        self.assertNotIn("double* gasGradientLimiter = nullptr;", cuda)
        limiter_kernel = cuda.split("computeGasGradientLimiterKernel", 1)[1].split(
            "computeGasEddyViscosityKernel", 1
        )[0]
        for limiter in ("rhoLimiter", "uxLimiter", "uyLimiter", "uzLimiter", "pLimiter", "tLimiter"):
            with self.subTest(limiter=limiter):
                self.assertIn(f"double {limiter} = 1.0;", limiter_kernel)
        self.assertNotIn("const double commonLimiter", limiter_kernel)
        limiter_by_field = {
            "gasGradientLimiterRho": "rhoLimiter",
            "gasGradientLimiterUx": "uxLimiter",
            "gasGradientLimiterUy": "uyLimiter",
            "gasGradientLimiterUz": "uzLimiter",
            "gasGradientLimiterP": "pLimiter",
            "gasGradientLimiterT": "tLimiter",
        }
        for field, limiter in limiter_by_field.items():
            with self.subTest(component_limiter_assignment=field):
                self.assertRegex(
                    limiter_kernel,
                    rf"s\.{field}\[c\]\s*=\s*"
                    rf"clampRange\(finiteOr\({limiter},\s*0\.0\),\s*"
                    rf"0\.0,\s*1\.0\);",
                )

        reconstruction = cuda.split("reconstructGasCellToFace", 1)[1].split(
            "molecularGasConductivity", 1
        )[0]
        for field in limiter_fields[:-1]:
            with self.subTest(reconstruction_limiter=field):
                self.assertIn(field, reconstruction)
        self.assertNotIn("gasGradientLimiterT", reconstruction)
        self.assertIn("makeGasPrimDevice", reconstruction)

        gradient = function_block(cuda, "computeGasPrimitiveGradientsKernel")
        boundary = function_block(cuda, "riemannFacePrimitiveForGradient")
        limiter = function_block(cuda, "computeGasGradientLimiterKernel")
        self.assertIn("riemannFacePrimitiveForGradient(s, c, f)", gradient)
        self.assertIn("riemannFacePrimitiveForGradient(s, c, f)", limiter)
        self.assertNotRegex(
            code_only(boundary + gradient + limiter),
            r"\bs\.gasBoundary[A-Z]",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
