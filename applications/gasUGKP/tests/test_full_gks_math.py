#!/usr/bin/env python3
                                                               

                                                                             
                                                                  

                                       

                                                                          
      

                                                               

                                                                         
                                                                             
                                           
   

from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

try:
    from .gks_reference import (
        MAX_INTERNAL_POWER,
        MAX_VELOCITY_ORDER,
        MaxwellState,
        ONE,
        PSI,
        apply_prandtl_energy_correction,
        build_gks_coefficients,
        coefficient_polynomial,
        coefficient_residual,
        compatible_interface_state,
        conservative_moments,
        euler_flux,
        flux_moments,
        full_normal_moments,
        full_second_order_gks_flux,
        gks_time_coefficients,
        integrate_polynomial,
        internal_degrees_from_gamma,
        internal_square_moments,
        matrix_vector,
        moment_matrix,
        molecular_heat_flux,
        molecular_viscous_flux,
        negative_half_normal_moments,
        polynomial_add,
        polynomial_multiply,
        polynomial_scale,
        positive_half_normal_moments,
        solve_5x5_partial_pivot,
        spatial_coefficient,
        spatial_transport_polynomial,
        temporal_coefficient,
        temporal_compatibility_residual,
    )
except ImportError:
    from gks_reference import (                          
        MAX_INTERNAL_POWER,
        MAX_VELOCITY_ORDER,
        MaxwellState,
        ONE,
        PSI,
        apply_prandtl_energy_correction,
        build_gks_coefficients,
        coefficient_polynomial,
        coefficient_residual,
        compatible_interface_state,
        conservative_moments,
        euler_flux,
        flux_moments,
        full_normal_moments,
        full_second_order_gks_flux,
        gks_time_coefficients,
        integrate_polynomial,
        internal_degrees_from_gamma,
        internal_square_moments,
        matrix_vector,
        moment_matrix,
        molecular_heat_flux,
        molecular_viscous_flux,
        negative_half_normal_moments,
        polynomial_add,
        polynomial_multiply,
        polynomial_scale,
        positive_half_normal_moments,
        solve_5x5_partial_pivot,
        spatial_coefficient,
        spatial_transport_polynomial,
        temporal_coefficient,
        temporal_compatibility_residual,
    )


ROOT = Path(__file__).resolve().parents[1]
ZERO5 = (0.0, 0.0, 0.0, 0.0, 0.0)
ZERO_GRADIENTS = (ZERO5, ZERO5, ZERO5)


def adaptive_simpson(
    function,
    lower: float,
    upper: float,
    absolute_tolerance: float = 2.0e-11,
    max_depth: int = 24,
) -> float:
                                                                           

    if lower == upper:
        return 0.0
    if lower > upper:
        return -adaptive_simpson(
            function, upper, lower, absolute_tolerance, max_depth
        )

    middle = 0.5 * (lower + upper)
    f_lower = function(lower)
    f_middle = function(middle)
    f_upper = function(upper)
    whole = (upper - lower) * (f_lower + 4.0 * f_middle + f_upper) / 6.0

    def recurse(a, b, fa, fm, fb, estimate, tolerance, depth):
        midpoint = 0.5 * (a + b)
        left_midpoint = 0.5 * (a + midpoint)
        right_midpoint = 0.5 * (midpoint + b)
        f_left_midpoint = function(left_midpoint)
        f_right_midpoint = function(right_midpoint)
        left = (midpoint - a) * (fa + 4.0 * f_left_midpoint + fm) / 6.0
        right = (b - midpoint) * (fm + 4.0 * f_right_midpoint + fb) / 6.0
        difference = left + right - estimate
        if depth <= 0 or abs(difference) <= 15.0 * tolerance:
            return left + right + difference / 15.0
        return recurse(
            a,
            midpoint,
            fa,
            f_left_midpoint,
            fm,
            left,
            tolerance / 2.0,
            depth - 1,
        ) + recurse(
            midpoint,
            b,
            fm,
            f_right_midpoint,
            fb,
            right,
            tolerance / 2.0,
            depth - 1,
        )

    return recurse(
        lower,
        upper,
        f_lower,
        f_middle,
        f_upper,
        whole,
        absolute_tolerance,
        max_depth,
    )


def numerical_normal_moment(
    mean: float, variance: float, order: int, half: str | None
) -> float:
    sigma = math.sqrt(variance)
    lower = mean - 12.0 * sigma
    upper = mean + 12.0 * sigma
    if half == "+":
        lower = max(0.0, lower)
        upper = max(0.0, upper)
    elif half == "-":
        lower = min(0.0, lower)
        upper = min(0.0, upper)
    elif half is not None:
        raise ValueError("invalid half")
    inverse_norm = 1.0 / math.sqrt(2.0 * math.pi * variance)

    def integrand(value: float) -> float:
        return (
            value**order
            * inverse_norm
            * math.exp(-0.5 * (value - mean) ** 2 / variance)
        )

    scale = max(1.0, abs(mean) ** order, variance ** (0.5 * order))
    return adaptive_simpson(integrand, lower, upper, 2.0e-12 * scale)


def numerical_internal_square_moment(k: float, rt: float, power: int) -> float:
    if k == 0.0:
        return 1.0 if power == 0 else 0.0
    shape = 0.5 * k
    scale = 2.0 * rt
    normalisation = math.gamma(shape)

    def integrand(y: float) -> float:
        if y == 0.0:
            exponent = shape + power - 1.0
            if abs(exponent) <= 1.0e-12:
                return scale**power / normalisation
            return 0.0 if exponent > 0.0 else math.inf
        return (
            scale**power
            * y ** (shape + power - 1.0)
            * math.exp(-y)
            / normalisation
        )

    return adaptive_simpson(integrand, 0.0, 60.0, 2.0e-11)


def assert_vector_close(
    case: unittest.TestCase,
    actual,
    expected,
    relative: float = 2.0e-11,
    absolute: float = 2.0e-11,
) -> None:
    case.assertEqual(len(actual), len(expected))
    for index, (got, want) in enumerate(zip(actual, expected)):
        case.assertTrue(
            math.isclose(got, want, rel_tol=relative, abs_tol=absolute),
            f"component {index}: got {got:.17g}, expected {want:.17g}",
        )


class MaxwellMomentReferenceTests(unittest.TestCase):
    def test_full_and_half_raw_moments_through_sixth_order(self) -> None:
        states = (
            (-0.8, 0.7),
            (0.0, 2.3),
            (1.1, 1.4),
        )
        for gamma in (1.2, 1.4):
                                                                          
                                                                         
            self.assertGreaterEqual(internal_degrees_from_gamma(gamma), 0.0)
            for mean, variance in states:
                analytic_full = full_normal_moments(mean, variance, MAX_VELOCITY_ORDER)
                analytic_positive = positive_half_normal_moments(
                    mean, variance, MAX_VELOCITY_ORDER
                )
                analytic_negative = negative_half_normal_moments(
                    mean, variance, MAX_VELOCITY_ORDER
                )
                for order in range(MAX_VELOCITY_ORDER + 1):
                    scale = max(1.0, abs(analytic_full[order]))
                    for half, analytic in (
                        (None, analytic_full[order]),
                        ("+", analytic_positive[order]),
                        ("-", analytic_negative[order]),
                    ):
                        with self.subTest(
                            gamma=gamma,
                            mean=mean,
                            variance=variance,
                            order=order,
                            half=half,
                        ):
                            numerical = numerical_normal_moment(
                                mean, variance, order, half
                            )
                            self.assertAlmostEqual(
                                analytic,
                                numerical,
                                delta=2.0e-8 * scale,
                            )
                    self.assertAlmostEqual(
                        analytic_positive[order] + analytic_negative[order],
                        analytic_full[order],
                        delta=2.0e-13 * scale,
                    )

    def test_internal_degree_moments_against_gamma_quadrature(self) -> None:
        for gamma in (1.2, 1.4):
            k = internal_degrees_from_gamma(gamma)
            for rt in (0.35, 2.7):
                analytic = internal_square_moments(k, rt, MAX_INTERNAL_POWER)
                for power in range(MAX_INTERNAL_POWER + 1):
                    with self.subTest(gamma=gamma, rt=rt, power=power):
                        numerical = numerical_internal_square_moment(k, rt, power)
                        self.assertAlmostEqual(
                            analytic[power],
                            numerical,
                            delta=2.0e-8 * max(1.0, abs(analytic[power])),
                        )

    def test_collision_invariants_recover_conservative_state(self) -> None:
        state = MaxwellState(1.7, 0.8, -0.35, 0.2, 2.4, 7.0)
        assert_vector_close(self, conservative_moments(state), state.conservative)
        halves = tuple(
            conservative_moments(state, half="+")[i]
            + conservative_moments(state, half="-")[i]
            for i in range(5)
        )
        assert_vector_close(self, halves, state.conservative)


class CoefficientSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = MaxwellState(1.4, 0.7, -0.2, 0.35, 1.8, 7.0)

    def test_scaled_partial_pivot_solve_recovers_five_coefficients(self) -> None:
        expected = (0.14, -0.08, 0.035, 0.21, -0.012)
        matrix = moment_matrix(self.state)
        rhs = matrix_vector(matrix, expected)
        recovered = solve_5x5_partial_pivot(matrix, rhs)
        assert_vector_close(self, recovered, expected, relative=2.0e-12)
        self.assertLess(coefficient_residual(self.state, recovered, rhs), 1.0e-12)

    def test_spatial_constraints_and_temporal_compatibility(self) -> None:
        derivatives = (
            (0.09, -0.04, 0.03, 0.02, 0.18),
            (-0.02, 0.06, -0.05, 0.01, -0.12),
            (0.04, 0.01, 0.02, -0.03, 0.07),
        )
        spatial = tuple(spatial_coefficient(self.state, value) for value in derivatives)
        for coefficient, derivative in zip(spatial, derivatives):
            self.assertLess(
                coefficient_residual(self.state, coefficient, derivative), 1.0e-12
            )
        temporal = temporal_coefficient(self.state, spatial)
        self.assertLess(
            temporal_compatibility_residual(self.state, spatial, temporal),
            1.0e-12,
        )

    def test_uniform_half_space_interface_recovers_the_same_maxwellian(self) -> None:
        interface = compatible_interface_state(self.state, self.state)
        assert_vector_close(self, interface.conservative, self.state.conservative)
        self.assertAlmostEqual(interface.rt, self.state.rt, places=13)

    def test_discontinuous_interface_and_half_derivatives_are_finite(self) -> None:
        left = MaxwellState(1.0, 1.2, 0.1, -0.2, 2.0, 2.0)
        right = MaxwellState(0.45, -0.25, -0.05, 0.1, 1.1, 2.0)
        left_gradients = (
            (0.03, 0.06, -0.01, 0.02, 0.15),
            ZERO5,
            (0.01, 0.01, 0.0, -0.01, 0.03),
        )
        right_gradients = (
            (-0.02, 0.01, 0.005, -0.01, -0.04),
            (0.0, 0.0, 0.01, 0.0, 0.0),
            ZERO5,
        )
        solved = build_gks_coefficients(left, right, left_gradients, right_gradients)
        self.assertGreater(solved.interface.rho, 0.0)
        self.assertGreater(solved.interface.rt, 0.0)
        for state, spatial, temporal in (
            (left, solved.left_spatial, solved.left_temporal),
            (right, solved.right_spatial, solved.right_temporal),
            (solved.interface, solved.interface_spatial, solved.interface_temporal),
        ):
            self.assertLess(
                temporal_compatibility_residual(state, spatial, temporal),
                1.0e-10,
            )


class CompleteTimeIntegrationTests(unittest.TestCase):
    def test_coefficients_match_direct_time_quadrature(self) -> None:
        for dt, tau in ((0.03, 0.2), (0.2, 0.03), (1.0e-7, 7.0)):
            with self.subTest(dt=dt, tau=tau):
                coefficients = gks_time_coefficients(dt, tau)
                equilibrium = adaptive_simpson(
                    lambda t: 1.0 - math.exp(-t / tau), 0.0, dt, 1.0e-15
                )
                free = adaptive_simpson(
                    lambda t: math.exp(-t / tau), 0.0, dt, 1.0e-15
                )
                time_exponential = adaptive_simpson(
                    lambda t: t * math.exp(-t / tau), 0.0, dt, 1.0e-15
                )
                self.assertAlmostEqual(coefficients.equilibrium, equilibrium, delta=3.0e-14)
                self.assertAlmostEqual(coefficients.free_transport, free, delta=3.0e-14)
                self.assertAlmostEqual(
                    coefficients.time_exponential,
                    time_exponential,
                    delta=3.0e-14,
                )
                self.assertAlmostEqual(coefficients.equilibrium_time, 0.5 * dt * dt)
                self.assertAlmostEqual(
                    coefficients.equilibrium + coefficients.free_transport,
                    dt,
                    delta=3.0e-14,
                )

    def test_small_and_large_relaxation_limits(self) -> None:
        dt = 0.1
        collision_dominated = gks_time_coefficients(dt, 1.0e-12).averaged(dt)
        self.assertAlmostEqual(collision_dominated.equilibrium, 1.0, delta=2.0e-11)
        self.assertAlmostEqual(collision_dominated.free_transport, 0.0, delta=2.0e-11)

        free_transport = gks_time_coefficients(dt, 1.0e8).averaged(dt)
        self.assertAlmostEqual(free_transport.equilibrium, 0.0, delta=2.0e-9)
        self.assertAlmostEqual(free_transport.free_transport, 1.0, delta=2.0e-9)
        self.assertAlmostEqual(free_transport.time_exponential, 0.5 * dt, delta=2.0e-9)

    def test_complete_flux_matches_direct_time_integration(self) -> None:
        left = MaxwellState(1.1, 0.65, -0.08, 0.12, 1.9, 2.0)
        right = MaxwellState(0.72, -0.15, 0.04, -0.06, 1.3, 2.0)
        left_gradients = (
            (0.04, 0.06, -0.01, 0.02, 0.12),
            (-0.01, 0.0, 0.03, 0.0, -0.02),
            (0.02, -0.01, 0.0, 0.01, 0.03),
        )
        right_gradients = (
            (-0.03, 0.02, 0.01, -0.01, -0.07),
            (0.01, -0.01, 0.0, 0.0, 0.02),
            ZERO5,
        )
        dt = 0.07
        numerical_tau = 0.045
        viscosity = 0.08
        analytic, solved, _ = full_second_order_gks_flux(
            left,
            right,
            left_gradients,
            right_gradients,
            dt,
            numerical_tau,
            viscosity,
            prandtl=1.0,
        )

        bar_spatial = spatial_transport_polynomial(solved.interface_spatial)
        left_spatial = spatial_transport_polynomial(solved.left_spatial)
        right_spatial = spatial_transport_polynomial(solved.right_spatial)
        bar_temporal = coefficient_polynomial(solved.interface_temporal)
        left_temporal = coefficient_polynomial(solved.left_temporal)
        right_temporal = coefficient_polynomial(solved.right_temporal)
        bar_ce = polynomial_add(bar_spatial, bar_temporal)
        left_ce = polynomial_add(left_spatial, left_temporal)
        right_ce = polynomial_add(right_spatial, right_temporal)
        physical_tau = viscosity / solved.interface.pressure

        def split(left_multiplier, right_multiplier):
            return tuple(
                flux_moments(left, left_multiplier, "+")[i]
                + flux_moments(right, right_multiplier, "-")[i]
                for i in range(5)
            )

        equilibrium_flux = flux_moments(solved.interface)
        free_flux = split(ONE, ONE)
        equilibrium_temporal = flux_moments(solved.interface, bar_temporal)
        equilibrium_ce = flux_moments(solved.interface, bar_ce)
        free_ce = split(left_ce, right_ce)
        equilibrium_spatial = flux_moments(solved.interface, bar_spatial)
        free_spatial = split(left_spatial, right_spatial)

        def instantaneous(component):
            def value(t):
                exponential = math.exp(-t / numerical_tau)
                return (
                    (1.0 - exponential) * equilibrium_flux[component]
                    + exponential * free_flux[component]
                    + t * equilibrium_temporal[component]
                    - physical_tau * (1.0 - exponential) * equilibrium_ce[component]
                    - physical_tau * exponential * free_ce[component]
                    + t
                    * exponential
                    * (equilibrium_spatial[component] - free_spatial[component])
                )

            return value

        numerical = tuple(
            adaptive_simpson(instantaneous(i), 0.0, dt, 2.0e-12) / dt
            for i in range(5)
        )
        assert_vector_close(self, analytic, numerical, relative=4.0e-10)

    def test_uniform_state_flux_is_exact_euler_flux(self) -> None:
        state = MaxwellState(1.3, 0.8, -0.3, 0.2, 2.1, 7.0)
        for numerical_tau, viscosity, prandtl in (
            (0.0, 0.0, 1.0),
            (0.02, 0.1, 0.4),
            (2.0, 0.7, 0.8),
        ):
            with self.subTest(
                numerical_tau=numerical_tau,
                viscosity=viscosity,
                prandtl=prandtl,
            ):
                flux, _, viscous = full_second_order_gks_flux(
                    state,
                    state,
                    ZERO_GRADIENTS,
                    ZERO_GRADIENTS,
                    dt=0.03,
                    numerical_tau=numerical_tau,
                    molecular_viscosity=viscosity,
                    prandtl=prandtl,
                )
                assert_vector_close(self, flux, euler_flux(state), relative=3.0e-12)
                assert_vector_close(self, viscous, ZERO5, absolute=3.0e-12)


class MolecularTransportAndPrandtlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gamma = 1.4
        self.k = internal_degrees_from_gamma(self.gamma)
        self.state = MaxwellState(1.2, 0.0, 0.0, 0.0, 2.5, self.k)
        self.mu = 0.3

    def _viscous_flux_from_normal_derivative(self, derivative):
        spatial = (
            spatial_coefficient(self.state, derivative),
            spatial_coefficient(self.state, ZERO5),
            spatial_coefficient(self.state, ZERO5),
        )
        temporal = temporal_coefficient(self.state, spatial)
        self.assertLess(
            temporal_compatibility_residual(self.state, spatial, temporal), 1.0e-12
        )
        return molecular_viscous_flux(
            self.state, spatial, temporal, self.mu / self.state.pressure
        )

    def test_manufactured_shear_recovers_newtonian_momentum_flux(self) -> None:
        tangential_gradient = 0.7
        derivative = (
            0.0,
            0.0,
            self.state.rho * tangential_gradient,
            0.0,
            0.0,
        )
        viscous = self._viscous_flux_from_normal_derivative(derivative)
        self.assertAlmostEqual(
            viscous[2], -self.mu * tangential_gradient, places=13
        )
        self.assertAlmostEqual(viscous[0], 0.0, places=13)
        self.assertAlmostEqual(viscous[4], 0.0, places=13)

    def test_constant_pressure_temperature_gradient_recovers_fourier_flux(self) -> None:
        d_rt_dx = 0.4
        d_rho_dx = -self.state.rho * d_rt_dx / self.state.rt
        viscous = self._viscous_flux_from_normal_derivative(
            (d_rho_dx, 0.0, 0.0, 0.0, 0.0)
        )
        cp_with_r_equal_one = 0.5 * (5.0 + self.k)
        expected_bgk_heat = -self.mu * cp_with_r_equal_one * d_rt_dx
        self.assertAlmostEqual(
            molecular_heat_flux(viscous, self.state), expected_bgk_heat, places=13
        )

    def test_prandtl_range_changes_only_energy_and_hits_target_conductivity(self) -> None:
        d_rt_dx = 0.4
        d_rho_dx = -self.state.rho * d_rt_dx / self.state.rt
        viscous = self._viscous_flux_from_normal_derivative(
            (d_rho_dx, 0.0, 0.0, 0.0, 0.0)
        )
        bgk_heat = molecular_heat_flux(viscous, self.state)
        baseline = (0.9, -0.3, 0.2, -0.1, 1.7)
        for prandtl in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            with self.subTest(prandtl=prandtl):
                corrected = apply_prandtl_energy_correction(
                    baseline, viscous, self.state, prandtl
                )
                self.assertEqual(corrected[:4], baseline[:4])
                self.assertAlmostEqual(
                    corrected[4] - baseline[4],
                    (1.0 / prandtl - 1.0) * bgk_heat,
                    places=14,
                )
                total_molecular_heat = bgk_heat + corrected[4] - baseline[4]
                self.assertAlmostEqual(total_molecular_heat, bgk_heat / prandtl, places=13)

    def test_prandtl_one_is_identity(self) -> None:
        baseline = (1.0, 2.0, 3.0, 4.0, 5.0)
        viscous = (0.0, -0.1, 0.2, -0.3, 0.7)
        self.assertEqual(
            apply_prandtl_energy_correction(baseline, viscous, self.state, 1.0),
            baseline,
        )


class ProductionCudaSourceGate(unittest.TestCase):
    def test_full_gks_cuda_source_contract(self) -> None:
        if os.environ.get("UGKP_REQUIRE_CUDA_GKS") != "1":
            self.skipTest(
                "CUDA full-GKS gate intentionally deferred during Task 2; "
                "set UGKP_REQUIRE_CUDA_GKS=1 to make it mandatory"
            )

        header = ROOT / "private_backend" / "FullSecondOrderGks.cuh"
        self.assertTrue(header.is_file(), "missing private_backend/FullSecondOrderGks.cuh")
        source = header.read_text(encoding="utf-8", errors="replace")
        self.assertRegex(
            source,
            r"#define\s+UGKP_FULL_SECOND_ORDER_GKS(?:\s+1)?(?:\s|$)",
            "the stable full-GKS implementation marker macro is missing",
        )
        for token in (
            "namespace ugkpgks",
            "solveFiveByFive",
            "timeCoefficients",
            "fullSecondOrderFlux",
            "molecularHeatFlux",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

        active_paths = (
            "createFields.H",
            "diluteUgkwpFoam.C",
            "gpu/GpuBackendApi.H",
            "gpu/GpuBackendClient.C",
            "gpu/GpuBackendProtocol.H",
            "gpu/GpuResidentStrict.H",
            "private_backend/GpuBackendServer.C",
            "private_backend/GpuResidentStrict.cu",
            "private_backend/FullSecondOrderGks.cuh",
        )
        gas_kappa_offenders = [
            relative
            for relative in active_paths
            if "gasKappa"
            in (ROOT / relative).read_text(encoding="utf-8", errors="replace")
        ]
        self.assertFalse(
            gas_kappa_offenders,
            "gasKappa remains in the executable interface or resident backend: "
            + ", ".join(gas_kappa_offenders),
        )
        cuda_source = (ROOT / "private_backend/GpuResidentStrict.cu").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertTrue(
            '#include "FullSecondOrderGks.cuh"' in cuda_source,
            "private_backend/GpuResidentStrict.cu does not include "
            "FullSecondOrderGks.cuh; the mathematical implementation is not "
            "wired into the resident face-flux path",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
