                                                                    

                                                                     
                                                                          
                                                                          
                          

                                                                           
                                                                            
                                                                          
                   
   

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


Vector5 = tuple[float, float, float, float, float]
Matrix5 = tuple[Vector5, Vector5, Vector5, Vector5, Vector5]
Exponent = tuple[int, int, int, int]
Polynomial = dict[Exponent, float]

ZERO_EXPONENT: Exponent = (0, 0, 0, 0)
MAX_VELOCITY_ORDER = 6
MAX_INTERNAL_POWER = 3


@dataclass(frozen=True)
class MaxwellState:
                                                                             

    rho: float
    u: float
    v: float
    w: float
    rt: float
    internal_degrees: float

    def __post_init__(self) -> None:
        values = (self.rho, self.u, self.v, self.w, self.rt, self.internal_degrees)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Maxwell state values must be finite")
        if self.rho <= 0.0 or self.rt <= 0.0:
            raise ValueError("rho and RT must be positive")
        if self.internal_degrees < 0.0:
            raise ValueError("internal degrees of freedom must be non-negative")

    @property
    def pressure(self) -> float:
        return self.rho * self.rt

    @property
    def conservative(self) -> Vector5:
        velocity_sq = self.u * self.u + self.v * self.v + self.w * self.w
        energy = 0.5 * self.rho * (
            velocity_sq + (3.0 + self.internal_degrees) * self.rt
        )
        return (
            self.rho,
            self.rho * self.u,
            self.rho * self.v,
            self.rho * self.w,
            energy,
        )


@dataclass(frozen=True)
class TimeCoefficients:
                                                                        

    equilibrium: float
    free_transport: float
    time_exponential: float
    equilibrium_time: float

    def averaged(self, dt: float) -> "TimeCoefficients":
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        return TimeCoefficients(
            self.equilibrium / dt,
            self.free_transport / dt,
            self.time_exponential / dt,
            self.equilibrium_time / dt,
        )


@dataclass(frozen=True)
class GksCoefficients:
                                                                          

    interface: MaxwellState
    left_spatial: tuple[Vector5, Vector5, Vector5]
    right_spatial: tuple[Vector5, Vector5, Vector5]
    interface_spatial: tuple[Vector5, Vector5, Vector5]
    left_temporal: Vector5
    right_temporal: Vector5
    interface_temporal: Vector5


def internal_degrees_from_gamma(gamma: float) -> float:
    if not math.isfinite(gamma) or not (1.0 < gamma <= 5.0 / 3.0):
        raise ValueError("gamma must satisfy 1 < gamma <= 5/3")
    return 2.0 / (gamma - 1.0) - 3.0


def full_normal_moments(mean: float, variance: float, order: int = 6) -> tuple[float, ...]:
                                                                 

    _validate_normal_arguments(mean, variance, order)
    moments = [1.0]
    if order:
        moments.append(mean)
    for n in range(2, order + 1):
        moments.append(mean * moments[n - 1] + (n - 1) * variance * moments[n - 2])
    return tuple(moments)


def positive_half_normal_moments(
    mean: float, variance: float, order: int = 6
) -> tuple[float, ...]:
                                                                          

    _validate_normal_arguments(mean, variance, order)
    sigma = math.sqrt(variance)
    beta = mean / (math.sqrt(2.0) * sigma)
    probability = 0.5 * math.erfc(-beta)
    boundary = sigma * math.exp(-(beta * beta)) / math.sqrt(2.0 * math.pi)
    moments = [probability]
    if order:
        moments.append(mean * probability + boundary)
    for n in range(2, order + 1):
        moments.append(mean * moments[n - 1] + (n - 1) * variance * moments[n - 2])
    return tuple(moments)


def negative_half_normal_moments(
    mean: float, variance: float, order: int = 6
) -> tuple[float, ...]:
                                                                               

    reflected = positive_half_normal_moments(-mean, variance, order)
    return tuple(((-1.0) ** n) * reflected[n] for n in range(order + 1))


def internal_square_moments(
    internal_degrees: float, rt: float, power: int = 3
) -> tuple[float, ...]:
                                                              

    if not math.isfinite(internal_degrees) or internal_degrees < 0.0:
        raise ValueError("internal degrees must be finite and non-negative")
    if not math.isfinite(rt) or rt <= 0.0:
        raise ValueError("RT must be finite and positive")
    if power < 0 or power > MAX_INTERNAL_POWER:
        raise ValueError(f"internal moment power must be in [0, {MAX_INTERNAL_POWER}]")
    moments = [1.0]
    for m in range(1, power + 1):
        moments.append(moments[-1] * rt * (internal_degrees + 2.0 * (m - 1)))
    return tuple(moments)


def _validate_normal_arguments(mean: float, variance: float, order: int) -> None:
    if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("normal mean must be finite and variance positive")
    if order < 0 or order > MAX_VELOCITY_ORDER:
        raise ValueError(f"normal moment order must be in [0, {MAX_VELOCITY_ORDER}]")


def polynomial_constant(value: float = 1.0) -> Polynomial:
    return {} if value == 0.0 else {ZERO_EXPONENT: float(value)}


def polynomial_monomial(exponent: Exponent, coefficient: float = 1.0) -> Polynomial:
    if len(exponent) != 4 or any(power < 0 for power in exponent):
        raise ValueError("a monomial needs four non-negative exponents")
    return {} if coefficient == 0.0 else {exponent: float(coefficient)}


def polynomial_add(*polynomials: Mapping[Exponent, float]) -> Polynomial:
    result: Polynomial = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, 0.0) + coefficient
    return {exponent: value for exponent, value in result.items() if value != 0.0}


def polynomial_scale(polynomial: Mapping[Exponent, float], scale: float) -> Polynomial:
    if scale == 0.0:
        return {}
    return {exponent: scale * value for exponent, value in polynomial.items()}


def polynomial_multiply(
    left: Mapping[Exponent, float], right: Mapping[Exponent, float]
) -> Polynomial:
    result: Polynomial = {}
    for a, left_value in left.items():
        for b, right_value in right.items():
            exponent = tuple(a[i] + b[i] for i in range(4))
            if max(exponent[:3]) > MAX_VELOCITY_ORDER or exponent[3] > MAX_INTERNAL_POWER:
                raise ValueError(f"GKS polynomial exceeds supported moment order: {exponent}")
            result[exponent] = result.get(exponent, 0.0) + left_value * right_value
    return {exponent: value for exponent, value in result.items() if value != 0.0}


ONE = polynomial_constant()
U_POLY = polynomial_monomial((1, 0, 0, 0))
V_POLY = polynomial_monomial((0, 1, 0, 0))
W_POLY = polynomial_monomial((0, 0, 1, 0))
Q_POLY = {
    (2, 0, 0, 0): 0.5,
    (0, 2, 0, 0): 0.5,
    (0, 0, 2, 0): 0.5,
    (0, 0, 0, 1): 0.5,
}
PSI: tuple[Polynomial, Polynomial, Polynomial, Polynomial, Polynomial] = (
    ONE,
    U_POLY,
    V_POLY,
    W_POLY,
    Q_POLY,
)


def coefficient_polynomial(coefficients: Sequence[float]) -> Polynomial:
    if len(coefficients) != 5:
        raise ValueError("a GKS derivative coefficient has five entries")
    return polynomial_add(
        *(polynomial_scale(PSI[i], float(coefficients[i])) for i in range(5))
    )


def maxwell_monomial_moment(
    state: MaxwellState, exponent: Exponent, half: str | None = None
) -> float:
                                                                         

    pu, pv, pw, pxi2 = exponent
    if max(pu, pv, pw) > MAX_VELOCITY_ORDER or pxi2 > MAX_INTERNAL_POWER:
        raise ValueError(f"moment order not supported: {exponent}")
    if half is None:
        normal = full_normal_moments(state.u, state.rt, pu)
    elif half == "+":
        normal = positive_half_normal_moments(state.u, state.rt, pu)
    elif half == "-":
        normal = negative_half_normal_moments(state.u, state.rt, pu)
    else:
        raise ValueError("half must be None, '+' or '-'")
    tangent_v = full_normal_moments(state.v, state.rt, pv)
    tangent_w = full_normal_moments(state.w, state.rt, pw)
    internal = internal_square_moments(state.internal_degrees, state.rt, pxi2)
    return state.rho * normal[pu] * tangent_v[pv] * tangent_w[pw] * internal[pxi2]


def integrate_polynomial(
    state: MaxwellState, polynomial: Mapping[Exponent, float], half: str | None = None
) -> float:
    return math.fsum(
        coefficient * maxwell_monomial_moment(state, exponent, half)
        for exponent, coefficient in polynomial.items()
    )


def conservative_moments(
    state: MaxwellState,
    multiplier: Mapping[Exponent, float] = ONE,
    half: str | None = None,
) -> Vector5:
    return tuple(
        integrate_polynomial(state, polynomial_multiply(PSI[i], multiplier), half)
        for i in range(5)
    )                              


def flux_moments(
    state: MaxwellState,
    multiplier: Mapping[Exponent, float] = ONE,
    half: str | None = None,
) -> Vector5:
    flux_multiplier = polynomial_multiply(U_POLY, multiplier)
    return conservative_moments(state, flux_multiplier, half)


def moment_matrix(state: MaxwellState) -> Matrix5:
    return tuple(
        tuple(integrate_polynomial(state, polynomial_multiply(PSI[i], PSI[j])) for j in range(5))
        for i in range(5)
    )                              


def matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector5:
    if len(matrix) != 5 or len(vector) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("matrix_vector requires a 5x5 matrix and a five-vector")
    return tuple(math.fsum(row[j] * vector[j] for j in range(5)) for row in matrix)                              


def solve_5x5_partial_pivot(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> Vector5:
                                                                             

    if len(matrix) != 5 or len(rhs) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("solve requires a 5x5 matrix and a five-vector")
    augmented = [[float(matrix[i][j]) for j in range(5)] + [float(rhs[i])] for i in range(5)]
    row_scales = [max(abs(value) for value in row[:5]) for row in augmented]
    if any(scale == 0.0 or not math.isfinite(scale) for scale in row_scales):
        raise ArithmeticError("singular or non-finite moment matrix")

    for column in range(5):
        pivot = max(
            range(column, 5),
            key=lambda row: abs(augmented[row][column]) / row_scales[row],
        )
        pivot_value = augmented[pivot][column]
        if not math.isfinite(pivot_value) or abs(pivot_value) <= 1.0e-15 * row_scales[pivot]:
            raise ArithmeticError("singular moment matrix")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            row_scales[column], row_scales[pivot] = row_scales[pivot], row_scales[column]

        for row in range(column + 1, 5):
            factor = augmented[row][column] / augmented[column][column]
            augmented[row][column] = 0.0
            for j in range(column + 1, 6):
                augmented[row][j] -= factor * augmented[column][j]

    solution = [0.0] * 5
    for row in range(4, -1, -1):
        numerator = augmented[row][5] - math.fsum(
            augmented[row][j] * solution[j] for j in range(row + 1, 5)
        )
        solution[row] = numerator / augmented[row][row]
    if not all(math.isfinite(value) for value in solution):
        raise ArithmeticError("non-finite coefficient solution")
    return tuple(solution)                              


def coefficient_residual(
    state: MaxwellState, coefficients: Sequence[float], target: Sequence[float]
) -> float:
    recovered = matrix_vector(moment_matrix(state), coefficients)
    numerator = math.sqrt(math.fsum((recovered[i] - target[i]) ** 2 for i in range(5)))
    denominator = max(
        math.sqrt(math.fsum(float(value) ** 2 for value in target)),
        state.rho,
        1.0,
    )
    return numerator / denominator


def spatial_coefficient(state: MaxwellState, conservative_derivative: Sequence[float]) -> Vector5:
    return solve_5x5_partial_pivot(moment_matrix(state), conservative_derivative)


def spatial_transport_polynomial(spatial: Sequence[Sequence[float]]) -> Polynomial:
    if len(spatial) != 3:
        raise ValueError("three local spatial coefficient vectors are required")
    return polynomial_add(
        polynomial_multiply(coefficient_polynomial(spatial[0]), U_POLY),
        polynomial_multiply(coefficient_polynomial(spatial[1]), V_POLY),
        polynomial_multiply(coefficient_polynomial(spatial[2]), W_POLY),
    )


def temporal_coefficient(state: MaxwellState, spatial: Sequence[Sequence[float]]) -> Vector5:
    transport = spatial_transport_polynomial(spatial)
    rhs = tuple(-value for value in conservative_moments(state, transport))
    return solve_5x5_partial_pivot(moment_matrix(state), rhs)


def temporal_compatibility_residual(
    state: MaxwellState, spatial: Sequence[Sequence[float]], temporal: Sequence[float]
) -> float:
    total = polynomial_add(spatial_transport_polynomial(spatial), coefficient_polynomial(temporal))
    residual = conservative_moments(state, total)
    scale = max(state.rho, abs(state.conservative[4]), 1.0)
    return math.sqrt(math.fsum(value * value for value in residual)) / scale


def state_from_conservative(conservative: Sequence[float], internal_degrees: float) -> MaxwellState:
    if len(conservative) != 5:
        raise ValueError("conservative state needs five entries")
    rho = float(conservative[0])
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("interface density must be positive")
    u, v, w = (float(conservative[i]) / rho for i in range(1, 4))
    kinetic = 0.5 * rho * (u * u + v * v + w * w)
    thermal_energy = float(conservative[4]) - kinetic
    rt = 2.0 * thermal_energy / (rho * (3.0 + internal_degrees))
    return MaxwellState(rho, u, v, w, rt, internal_degrees)


def compatible_interface_state(left: MaxwellState, right: MaxwellState) -> MaxwellState:
    _check_matching_internal_degrees(left, right)
    left_half = conservative_moments(left, half="+")
    right_half = conservative_moments(right, half="-")
    conservative = tuple(left_half[i] + right_half[i] for i in range(5))
    return state_from_conservative(conservative, left.internal_degrees)


def _check_matching_internal_degrees(left: MaxwellState, right: MaxwellState) -> None:
    if not math.isclose(
        left.internal_degrees, right.internal_degrees, rel_tol=0.0, abs_tol=1.0e-13
    ):
        raise ValueError("left and right Maxwellians must use the same internal degrees")


def build_gks_coefficients(
    left: MaxwellState,
    right: MaxwellState,
    left_conservative_gradients: Sequence[Sequence[float]],
    right_conservative_gradients: Sequence[Sequence[float]],
) -> GksCoefficients:
                                                                           

    _check_matching_internal_degrees(left, right)
    if len(left_conservative_gradients) != 3 or len(right_conservative_gradients) != 3:
        raise ValueError("three local conservative gradients are required per side")
    left_spatial = tuple(
        spatial_coefficient(left, derivative) for derivative in left_conservative_gradients
    )
    right_spatial = tuple(
        spatial_coefficient(right, derivative) for derivative in right_conservative_gradients
    )
    interface = compatible_interface_state(left, right)

    interface_spatial_list: list[Vector5] = []
    for direction in range(3):
        left_poly = coefficient_polynomial(left_spatial[direction])
        right_poly = coefficient_polynomial(right_spatial[direction])
        derivative = tuple(
            conservative_moments(left, left_poly, "+")[i]
            + conservative_moments(right, right_poly, "-")[i]
            for i in range(5)
        )
        interface_spatial_list.append(spatial_coefficient(interface, derivative))
    interface_spatial = tuple(interface_spatial_list)

    return GksCoefficients(
        interface=interface,
        left_spatial=left_spatial,                          
        right_spatial=right_spatial,                          
        interface_spatial=interface_spatial,                          
        left_temporal=temporal_coefficient(left, left_spatial),
        right_temporal=temporal_coefficient(right, right_spatial),
        interface_temporal=temporal_coefficient(interface, interface_spatial),
    )


def _series_sum_x_minus_one_minus_exp(x: float) -> float:
                                               
    term = 0.5 * x * x
    total = term
    for n in range(3, 13):
        term *= -x / n
        total += term
    return total


def _series_sum_one_minus_exp_minus_x_exp(x: float) -> float:
                                                        
    power = x * x
    factorial = 2.0
    total = 0.5 * power
    for n in range(3, 13):
        power *= x
        factorial *= n
        total += ((-1.0) ** n) * (n - 1) * power / factorial
    return total


def gks_time_coefficients(dt: float, numerical_tau: float) -> TimeCoefficients:
                                                                              

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(numerical_tau) or numerical_tau < 0.0:
        raise ValueError("numerical tau must be finite and non-negative")
    if numerical_tau == 0.0:
        return TimeCoefficients(dt, 0.0, 0.0, 0.5 * dt * dt)

    x = dt / numerical_tau
    if x < 1.0e-4:
        ce = numerical_tau * _series_sum_x_minus_one_minus_exp(x)
        one_minus_exp = -math.expm1(-x)
        cf = numerical_tau * one_minus_exp
        ct = numerical_tau * numerical_tau * _series_sum_one_minus_exp_minus_x_exp(x)
    elif x > 745.0:
        ce = dt - numerical_tau
        cf = numerical_tau
        ct = numerical_tau * numerical_tau
    else:
        exponential = math.exp(-x)
        one_minus_exp = -math.expm1(-x)
        ce = dt - numerical_tau * one_minus_exp
        cf = numerical_tau * one_minus_exp
        ct = numerical_tau * numerical_tau * one_minus_exp - numerical_tau * dt * exponential
    return TimeCoefficients(ce, cf, ct, 0.5 * dt * dt)


def _vector_add(*vectors: Sequence[float]) -> Vector5:
    return tuple(math.fsum(vector[i] for vector in vectors) for i in range(5))                              


def _vector_scale(vector: Sequence[float], scale: float) -> Vector5:
    return tuple(scale * float(value) for value in vector)                              


def _split_flux(
    left: MaxwellState,
    right: MaxwellState,
    left_multiplier: Mapping[Exponent, float],
    right_multiplier: Mapping[Exponent, float],
) -> Vector5:
    return _vector_add(
        flux_moments(left, left_multiplier, "+"),
        flux_moments(right, right_multiplier, "-"),
    )


def molecular_viscous_flux(
    interface: MaxwellState,
    interface_spatial: Sequence[Sequence[float]],
    interface_temporal: Sequence[float],
    physical_tau: float,
) -> Vector5:
    if physical_tau < 0.0 or not math.isfinite(physical_tau):
        raise ValueError("physical tau must be finite and non-negative")
    ce_polynomial = polynomial_add(
        spatial_transport_polynomial(interface_spatial),
        coefficient_polynomial(interface_temporal),
    )
    return _vector_scale(flux_moments(interface, ce_polynomial), -physical_tau)


def molecular_heat_flux(
    viscous_flux: Sequence[float], interface: MaxwellState
) -> float:
    return (
        float(viscous_flux[4])
        - interface.u * float(viscous_flux[1])
        - interface.v * float(viscous_flux[2])
        - interface.w * float(viscous_flux[3])
    )


def apply_prandtl_energy_correction(
    flux: Sequence[float],
    viscous_flux: Sequence[float],
    interface: MaxwellState,
    prandtl: float,
) -> Vector5:
    if not math.isfinite(prandtl) or prandtl <= 0.0:
        raise ValueError("Prandtl number must be finite and positive")
    corrected = [float(value) for value in flux]
    corrected[4] += (1.0 / prandtl - 1.0) * molecular_heat_flux(
        viscous_flux, interface
    )
    return tuple(corrected)                              


def full_second_order_gks_flux(
    left: MaxwellState,
    right: MaxwellState,
    left_conservative_gradients: Sequence[Sequence[float]],
    right_conservative_gradients: Sequence[Sequence[float]],
    dt: float,
    numerical_tau: float,
    molecular_viscosity: float,
    prandtl: float = 1.0,
) -> tuple[Vector5, GksCoefficients, Vector5]:
                                                                             

    if molecular_viscosity < 0.0 or not math.isfinite(molecular_viscosity):
        raise ValueError("molecular viscosity must be finite and non-negative")
    coefficients = build_gks_coefficients(
        left, right, left_conservative_gradients, right_conservative_gradients
    )
    time = gks_time_coefficients(dt, numerical_tau)
    interface = coefficients.interface
    physical_tau = molecular_viscosity / interface.pressure

    bar_spatial = spatial_transport_polynomial(coefficients.interface_spatial)
    left_spatial = spatial_transport_polynomial(coefficients.left_spatial)
    right_spatial = spatial_transport_polynomial(coefficients.right_spatial)
    bar_temporal = coefficient_polynomial(coefficients.interface_temporal)
    left_temporal = coefficient_polynomial(coefficients.left_temporal)
    right_temporal = coefficient_polynomial(coefficients.right_temporal)
    bar_ce = polynomial_add(bar_spatial, bar_temporal)
    left_ce = polynomial_add(left_spatial, left_temporal)
    right_ce = polynomial_add(right_spatial, right_temporal)

    integrated = _vector_add(
        _vector_scale(flux_moments(interface), time.equilibrium),
        _vector_scale(_split_flux(left, right, ONE, ONE), time.free_transport),
        _vector_scale(flux_moments(interface, bar_temporal), time.equilibrium_time),
        _vector_scale(flux_moments(interface, bar_ce), -physical_tau * time.equilibrium),
        _vector_scale(
            _split_flux(left, right, left_ce, right_ce),
            -physical_tau * time.free_transport,
        ),
        _vector_scale(
            _vector_add(
                flux_moments(interface, bar_spatial),
                _vector_scale(_split_flux(left, right, left_spatial, right_spatial), -1.0),
            ),
            time.time_exponential,
        ),
    )
    averaged = _vector_scale(integrated, 1.0 / dt)
    viscous = molecular_viscous_flux(
        interface,
        coefficients.interface_spatial,
        coefficients.interface_temporal,
        physical_tau,
    )
    corrected = apply_prandtl_energy_correction(
        averaged, viscous, interface, prandtl
    )
    return corrected, coefficients, viscous


def euler_flux(state: MaxwellState) -> Vector5:
    rho, rho_u, rho_v, rho_w, rho_e = state.conservative
    return (
        rho_u,
        rho_u * state.u + state.pressure,
        rho_v * state.u,
        rho_w * state.u,
        (rho_e + state.pressure) * state.u,
    )
