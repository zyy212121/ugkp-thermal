import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"
MIRROR_HEADER = ROOT / "gpu" / "GpuResidentStrict.H"
PROTOCOL_HEADER = ROOT / "gpu" / "GpuBackendProtocol.H"
PACKING_ALGEBRA = (
    ROOT.parents[1] / "common" / "gasNumerics" /
    "GpuPackingProjectionAlgebra.cuh"
)


def braced_block_after(source: str, marker_pattern: str) -> str:
    marker = re.search(marker_pattern, source)
    if marker is None:
        raise AssertionError(f"missing block marker: {marker_pattern}")
    opening = source.find("{", marker.end())
    if opening < 0:
        raise AssertionError(f"missing opening brace after: {marker_pattern}")
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError(f"unclosed block after: {marker_pattern}")


def projected_jacobi_two_cell(rhs0: float, rhs1: float, iterations: int = 256):
                                                                                 
    p0 = 0.0
    p1 = 0.0
    for _ in range(iterations):
        candidate0 = max(0.0, rhs0 + p1)
        candidate1 = max(0.0, 0.5 * (rhs1 + p0))
        next0 = 0.2 * p0 + 0.8 * candidate0
        next1 = 0.2 * p1 + 0.8 * candidate1
        p0, p1 = next0, next1
    return p0, p1


def projected_jacobi_chain(rhs, iterations, active_cells=None):
    count = len(rhs)
    active = set(range(count)) if active_cells is None else set(active_cells)
    pressure = [0.0] * count
    for _ in range(iterations):
        next_pressure = pressure.copy()
        for cell in active:
            neighbours = []
            if cell > 0:
                neighbours.append(cell - 1)
            if cell + 1 < count:
                neighbours.append(cell + 1)
            diagonal = len(neighbours) + int(cell == count - 1)
            candidate = max(
                0.0,
                (rhs[cell] + sum(pressure[n] for n in neighbours)) / diagonal,
            )
            next_pressure[cell] = 0.2 * pressure[cell] + 0.8 * candidate
        pressure = next_pressure
    return pressure


def graph_halo(seed_cells, layers, cell_count):
    active = set(seed_cells)
    frontier = set(seed_cells)
    for _ in range(layers):
        next_frontier = set()
        for cell in frontier:
            if cell > 0 and cell - 1 not in active:
                next_frontier.add(cell - 1)
            if cell + 1 < cell_count and cell + 1 not in active:
                next_frontier.add(cell + 1)
        active.update(next_frontier)
        frontier = next_frontier
    return active


class MobilePackingProjectionMathTests(unittest.TestCase):
    def test_signed_underpacked_slack_localises_pressure(self):
        p0, p1 = projected_jacobi_two_cell(0.07, -0.07)
        self.assertAlmostEqual(p0, 0.07, places=12)
        self.assertAlmostEqual(p1, 0.0, places=12)

    def test_two_cell_correction_satisfies_constraint_and_complementarity(self):
        alpha_max = 0.63
        alpha_pred = (0.70, 0.56)
        rhs = tuple(alpha - alpha_max for alpha in alpha_pred)
        p0, p1 = projected_jacobi_two_cell(*rhs)

                                                                          
                                                                             
        q_owner = p0 - p1
        q_neighbour = -q_owner
        corrected = (alpha_pred[0] - q_owner, alpha_pred[1] - q_neighbour)
        self.assertAlmostEqual(corrected[0], alpha_max, places=12)
        self.assertAlmostEqual(corrected[1], alpha_max, places=12)

        residual = (p0 - p1 - rhs[0], 2.0 * p1 - p0 - rhs[1])
        self.assertGreaterEqual(min(residual), -1.0e-12)
        self.assertAlmostEqual(p0 * residual[0], 0.0, places=12)
        self.assertAlmostEqual(p1 * residual[1], 0.0, places=12)

    def test_face_fraction_prevents_dilute_receiver_division(self):
        solid_volume_flux = 0.07
        face_fraction = 0.5 * (0.63 + 1.0e-4)
        receiver_fraction = 1.0e-4
        face_velocity_flux = solid_volume_flux / face_fraction
        old_dilute_cell_scaling = solid_volume_flux / receiver_fraction
        self.assertLess(face_velocity_flux, 0.23)
        self.assertGreater(old_dilute_cell_scaling / face_velocity_flux, 3000.0)

    def test_no_excess_produces_zero_pressure(self):
        self.assertEqual(projected_jacobi_two_cell(0.0, 0.0), (0.0, 0.0))

    def test_iteration_depth_halo_matches_dense_fixed_iteration_result(self):
        rhs = [-0.20] * 31
        rhs[15] = 0.40
        iterations = 7
        active = graph_halo({15}, iterations, len(rhs))
        dense = projected_jacobi_chain(rhs, iterations)
        local = projected_jacobi_chain(rhs, iterations, active)
        self.assertEqual(local, dense)
        self.assertTrue(all(value == 0.0 for value in local[:8]))
        self.assertTrue(all(value == 0.0 for value in local[23:]))

    def test_correction_region_adds_exactly_one_graph_layer(self):
        pressure_active = graph_halo({8}, 3, 17)
        correction_active = graph_halo(pressure_active, 1, 17)
        self.assertEqual(correction_active - pressure_active, {4, 12})


class MobilePackingProjectionSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.header = MIRROR_HEADER.read_text(encoding="utf-8")
        cls.packing_algebra = PACKING_ALGEBRA.read_text(encoding="utf-8")

    def test_singular_jamming_equation_of_state_is_removed(self):
        self.assertNotIn("jammingPressureFromEpsilonDevice", self.source)
        self.assertNotIn("jammingPressureScale*activation2", self.source)
        self.assertNotIn("const double pTotal = pColl + pJam", self.source)

    def test_mobile_moment_accumulation_excludes_stuck_particles(self):
        start = self.source.index("accumulateMobilePackingMomentsKernel")
        end = self.source.index("normalizeMobilePackingMomentsKernel", start)
        body = self.source[start:end]
        self.assertIn("s.pStuck[i] != 0", body)
        self.assertIn("continue;", body)
        self.assertIn("s.mobilePackingRho", body)

    def test_final_particle_velocity_writeback_excludes_stuck_particles(self):
        kernel = braced_block_after(
            self.source,
            r"\b__global__\s+void\s+"
            r"applyMobilePackingCorrectionToParticlesKernel\s*\(",
        )
        particle_loop = braced_block_after(kernel, r"\bfor\s*\(")
        stuck_guard_pattern = (
            r"\bif\s*\(\s*(?:"
            r"s\.pStatus\[i\]\s*!=\s*1\s*\|\|\s*"
            r"s\.pStuck\[i\]\s*!=\s*0"
            r"|"
            r"s\.pStuck\[i\]\s*!=\s*0\s*\|\|\s*"
            r"s\.pStatus\[i\]\s*!=\s*1"
            r")\s*\)"
        )
        guard = re.search(stuck_guard_pattern, particle_loop)
        self.assertIsNotNone(guard)
        guard_block = braced_block_after(particle_loop, stuck_guard_pattern)
        self.assertRegex(guard_block, r"\A\s*continue\s*;\s*\Z")

        velocity_writes = tuple(
            f"s.{field}[i] ="
            for field in ("pux", "puy", "puz", "puxOld", "puyOld", "puzOld")
        )
        for write in velocity_writes:
            with self.subTest(write=write):
                self.assertIn(write, particle_loop)
                self.assertLess(guard.start(), particle_loop.index(write))

    def test_projection_is_cell_iterative_and_particle_single_pass(self):
        self.assertIn("solveActiveMobilePackingPressureJacobiKernel", self.source)
        self.assertNotIn("solveMobilePackingPressureFusedKernel", self.source)
        self.assertNotIn("mobilePackingSingleBlockCellLimit", self.source)
        self.assertIn("computeMobilePackingFaceCorrectionFluxKernel", self.source)
        self.assertIn("reconstructMobilePackingVelocityCorrectionKernel", self.source)
        self.assertIn("applyMobilePackingCorrectionToParticlesKernel", self.source)
        self.assertIn("int applyMobilePackingProjection", self.source)
        projection = self.source[
            self.source.index("int applyMobilePackingProjection") :
            self.source.index("__device__ double granularCollisionTauFromCellDevice")
        ]
        self.assertIn(
            "for (int iter = 0; iter < s->packingProjectionIterations; ++iter)",
            projection,
        )
        self.assertRegex(
            projection,
            r"solveActiveMobilePackingPressureJacobiKernel\s*"
            r"<<<activeGrid, block>>>",
        )
        self.assertNotIn("if (s->nCells <=", projection)
        self.assertNotIn("trackParticlesLocalFaceWalkKernel", projection)
        self.assertEqual(
            projection.count("applyMobilePackingCorrectionToParticlesKernel"),
            1,
        )

    def test_projection_iteration_count_is_runtime_configurable(self):
        self.assertIn("gpuResidentPackingProjectionIterations", self.header)
        self.assertIn("label(20)", self.header)
        self.assertIn("packingProjectionIterations", self.source)
        self.assertNotIn(
            "constexpr int mobilePackingProjectionIterations = 20",
            self.source,
        )

    def test_projection_builds_fixed_local_support_and_zero_seed_fast_path(self):
        projection = braced_block_after(
            self.source,
            r"\bint\s+applyMobilePackingProjection\s*\(",
        )
        self.assertIn("seedMobilePackingActivity", self.source)
        self.assertIn("expandMobilePackingActivityFrontierKernel", projection)
        self.assertIn(
            "layer < s->packingProjectionIterations",
            projection,
        )
        self.assertIn("if (seedCount == 0)", projection)
        zero_seed_block = braced_block_after(
            projection,
            r"\bif\s*\(\s*seedCount\s*==\s*0\s*\)",
        )
        self.assertRegex(zero_seed_block, r"\A\s*return\s+0\s*;\s*\Z")
        self.assertIn("pressureActiveCount", projection)
        self.assertIn("correctionActiveCount", projection)
        self.assertNotIn("activeFraction", projection)
        self.assertNotIn("denseFallback", projection)

    def test_local_projection_has_one_correction_halo(self):
        projection = braced_block_after(
            self.source,
            r"\bint\s+applyMobilePackingProjection\s*\(",
        )
        expansion = "expandMobilePackingActivityFrontierKernel"
        pressure_loop = braced_block_after(
            projection,
            r"for\s*\(\s*int\s+layer\s*=\s*0\s*;\s*"
            r"layer\s*<\s*s->packingProjectionIterations\s*;\s*"
            r"\+\+layer\s*\)",
        )
        self.assertEqual(pressure_loop.count(expansion), 1)

        correction_begin = projection.index(
            "initialiseMobilePackingCorrectionRegionKernel"
        )
        correction_end = projection.index(
            "int correctionActiveCount = 0;",
            correction_begin,
        )
        correction_setup = projection[correction_begin:correction_end]
        self.assertEqual(correction_setup.count(expansion), 1)
        self.assertNotRegex(correction_setup, r"\b(?:for|while)\s*\(")
        self.assertEqual(projection.count(expansion), 2)
        self.assertIn("mobilePackingCorrectionCellMask", self.source)
        self.assertIn("mobilePackingActiveCellMask", self.source)
        self.assertIn("mobilePackingCorrectionCellList", self.source)

    def test_projection_uses_signed_slack_in_the_lcp_rhs(self):
        start = self.source.index("prepareMobilePackingProjectionKernel")
        end = self.source.index(
            "solveActiveMobilePackingPressureJacobiKernel",
            start,
        )
        body = self.source[start:end]
        self.assertIn("epsPred - s.packingFraction", body)
        self.assertNotIn("fmax(epsPred - s.packingFraction", body)

        body = self.packing_algebra
        self.assertIn("finiteOr(s.pressureDeltaEnergy[c], 0.0)", body)
        self.assertNotIn(
            "clampMin(finiteOr(s.pressureDeltaEnergy[c], 0.0), 0.0)",
            body,
        )

    def test_pressure_maps_through_conservative_face_flux_not_cell_density(self):
        self.assertNotIn("computeMobilePackingCorrectionKernel", self.source)
        start = self.source.index("computeMobilePackingFaceCorrectionFluxKernel")
        end = self.source.index("reconstructMobilePackingVelocityCorrectionKernel", start)
        face_body = self.source[start:end]
        self.assertIn("solidVolumeFlux", face_body)
        self.assertIn("faceMobileFraction", face_body)
        self.assertIn("velocityCorrectionFlux", face_body)

        start = self.source.index("reconstructMobilePackingVelocityCorrectionKernel")
        end = self.source.index("applyMobilePackingCorrectionToParticlesKernel", start)
        reconstruction = self.source[start:end]
        self.assertIn("m00 += nx*s.Sfx[f];", reconstruction)
        self.assertIn("b0 += nx*velocityCorrectionFlux;", reconstruction)
        self.assertNotIn("-dt/(rho*", reconstruction)

    def test_particle_mapping_recovers_face_normal_flux(self):
        body = self.packing_algebra
        self.assertIn("closestPlane", body)
        self.assertIn("centreToFaceDistance", body)
        self.assertIn("particleToFaceDistance", body)
        self.assertIn("faceVelocityCorrection", body)

    def test_projection_runs_after_relaxation_and_before_tracking(self):
        relax = self.source.index("UGKP_DEV_PROBE_LEAVE(ProbeRelax)")
        project = self.source.index("applyMobilePackingProjection", relax)
        track = self.source.index("trackParticlesLocalFaceWalkKernel", project)
        self.assertLess(relax, project)
        self.assertLess(project, track)

    def test_collisional_pressure_kick_no_longer_owns_jamming(self):
        start = self.source.index("int applyCollisionalPressureKick")
        end = self.source.index("int applyMobilePackingProjection", start)
        body = self.source[start:end]
        self.assertIn("if (!s->collisionalPressureEnabled)", body)
        self.assertNotIn("jammingPressureEnabled", body)

    def test_frontend_keeps_controls_without_unsolicited_startup_report(self):
        self.assertIn("gpuResidentJammingPressure", self.header)
        self.assertIn("gpuResidentPackingFraction", self.header)
        self.assertNotIn("Particle packing projection:", self.header)
        self.assertNotIn("legacy jamming-pressure entries are ignored", self.header)

    def test_protocol_minor_marks_semantic_change(self):
        protocol = PROTOCOL_HEADER.read_text(encoding="utf-8")
        self.assertIn("protocolMajor = 6", protocol)
        self.assertIn("protocolMinor = 0", protocol)


if __name__ == "__main__":
    unittest.main()
