                                                                                 

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


_ROUND_OFF_FACTOR = 1.0e-12
_SMALL = 1.0e-300


class MieError(RuntimeError):
    pass


@dataclass(frozen=True)
class MieSolution:
    x: float
    qext: float
    qsca: float
    qabs: float
    g: float
    an: np.ndarray
    bn: np.ndarray

    @property
    def n_stop(self) -> int:
        return len(self.an) - 1


def _roundoff_nonnegative(value: float, scale: float, name: str) -> float:
    if not math.isfinite(value):
        raise MieError(f"non-finite {name}: {value}")
    tolerance = _ROUND_OFF_FACTOR * max(scale, 1.0)
    if value < -tolerance:
        raise MieError(f"materially negative {name}: {value} (tolerance {tolerance})")
    return 0.0 if value < 0.0 else value


def _coefficients_with_padding(
    x: float, refractive_index: complex, padding: int
) -> MieSolution:
    if not math.isfinite(x) or x <= 0.0:
        raise MieError(f"size parameter must be finite and positive, got {x}")
    if not (
        math.isfinite(refractive_index.real)
        and math.isfinite(refractive_index.imag)
        and abs(refractive_index) > 0.0
    ):
        raise MieError(f"invalid refractive index {refractive_index}")

    n_stop = max(1, int(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    n_max = max(n_stop, int(abs(refractive_index * x))) + padding
    logarithmic_derivative = np.zeros(n_max + 2, dtype=np.complex128)
    z = refractive_index * x
    for n in range(n_max, 0, -1):
        n_over_z = n / z
        logarithmic_derivative[n - 1] = n_over_z - 1.0 / (
            logarithmic_derivative[n] + n_over_z
        )

    an = np.zeros(n_stop + 1, dtype=np.complex128)
    bn = np.zeros(n_stop + 1, dtype=np.complex128)
    psi_nm1 = math.sin(x)
    psi_n = math.sin(x) / x - math.cos(x)
    xi_nm1 = complex(psi_nm1, math.cos(x))
    xi_n = complex(psi_n, math.cos(x) / x + math.sin(x))

    qext_sum = 0.0
    qsca_sum = 0.0
    for n in range(1, n_stop + 1):
        rn = float(n)
        alpha = logarithmic_derivative[n] / refractive_index + rn / x
        beta = refractive_index * logarithmic_derivative[n] + rn / x
        an[n] = (alpha * psi_n - psi_nm1) / (alpha * xi_n - xi_nm1)
        bn[n] = (beta * psi_n - psi_nm1) / (beta * xi_n - xi_nm1)
        factor = float(2 * n + 1)
        qext_sum += factor * (an[n] + bn[n]).real
        qsca_sum += factor * (abs(an[n]) ** 2 + abs(bn[n]) ** 2)

        psi_np1 = ((2 * n + 1) / x) * psi_n - psi_nm1
        xi_np1 = ((2 * n + 1) / x) * xi_n - xi_nm1
        psi_nm1, psi_n = psi_n, psi_np1
        xi_nm1, xi_n = xi_n, xi_np1

    qext = _roundoff_nonnegative(2.0 * qext_sum / (x * x), 1.0, "Qext")
    qsca = _roundoff_nonnegative(2.0 * qsca_sum / (x * x), qext, "Qsca")
    qabs = _roundoff_nonnegative(qext - qsca, qext, "Qabs")

    g = 0.0
    if qsca > _SMALL:
        g_sum = 0.0
        for n in range(1, n_stop):
            rn = float(n)
            g_sum += rn * (rn + 2.0) / (rn + 1.0) * (
                (an[n] * np.conj(an[n + 1])).real
                + (bn[n] * np.conj(bn[n + 1])).real
            )
        for n in range(1, n_stop + 1):
            rn = float(n)
            g_sum += (2.0 * rn + 1.0) / (rn * (rn + 1.0)) * (
                an[n] * np.conj(bn[n])
            ).real
        g = 4.0 * g_sum / (x * x * qsca)
        if not math.isfinite(g) or g < -1.0 - 1.0e-12 or g > 1.0 + 1.0e-12:
            raise MieError(f"invalid coefficient asymmetry factor g={g}")
        g = min(1.0, max(-1.0, g))

    return MieSolution(x, qext, qsca, qabs, g, an, bn)


def _solution_relative_change(previous: MieSolution, current: MieSolution) -> float:
    scalar_change = max(
        abs(new - old) / max(abs(new), abs(old), 1.0e-300)
        for old, new in (
            (previous.qext, current.qext),
            (previous.qsca, current.qsca),
            (previous.qabs, current.qabs),
            (previous.g, current.g),
        )
    )
    coefficient_scale = max(
        float(np.max(np.abs(previous.an))),
        float(np.max(np.abs(previous.bn))),
        float(np.max(np.abs(current.an))),
        float(np.max(np.abs(current.bn))),
        1.0e-300,
    )
    coefficient_change = max(
        float(np.max(np.abs(current.an - previous.an))),
        float(np.max(np.abs(current.bn - previous.bn))),
    ) / coefficient_scale
    return max(scalar_change, coefficient_change)


def coefficients(
    x: float,
    refractive_index: complex,
    *,
    convergence_rel_tolerance: float = 1.0e-10,
    initial_padding: int = 15,
    max_padding: int = 1920,
) -> MieSolution:
                                                                               
    if not math.isfinite(convergence_rel_tolerance) or convergence_rel_tolerance <= 0.0:
        raise MieError("radial convergence tolerance must be finite and positive")
    if (
        isinstance(initial_padding, bool)
        or isinstance(max_padding, bool)
        or int(initial_padding) != initial_padding
        or int(max_padding) != max_padding
        or initial_padding < 1
        or max_padding < initial_padding
    ):
        raise MieError("radial padding bounds must be positive integers in order")

    padding = int(initial_padding)
    previous = None
    while True:
        current = _coefficients_with_padding(x, refractive_index, padding)
        if previous is not None:
            relative_change = _solution_relative_change(previous, current)
            if relative_change <= convergence_rel_tolerance:
                return current
        if padding >= max_padding:
            raise MieError(
                "Mie radial recurrence did not converge: "
                f"x={x}, padding={padding}, maxPadding={max_padding}, "
                f"relativeChange={relative_change if previous is not None else math.inf:.6e}, "
                f"required<={convergence_rel_tolerance:.6e}"
            )
        previous = current
        padding = min(2 * padding, int(max_padding))


def angular_functions(mu_values, n_stop: int) -> Tuple[list, list]:
                                                                              
    mu = np.asarray(mu_values, dtype=float)
    if n_stop < 1:
        raise ValueError("n_stop must be at least one")
    pi_values = [np.zeros_like(mu) for _ in range(n_stop + 1)]
    tau_values = [np.zeros_like(mu) for _ in range(n_stop + 1)]
    pi_nm1 = np.zeros_like(mu)        
    pi_n = np.ones_like(mu)        
    for n in range(1, n_stop + 1):
        rn = float(n)
        pi_values[n] = pi_n.copy()
                                                                
        tau_values[n] = rn * mu * pi_n - float(n + 1) * pi_nm1
        pi_np1 = (
            (float(2 * n + 1) / rn) * mu * pi_n
            - (float(n + 1) / rn) * pi_nm1
        )
        pi_nm1, pi_n = pi_n, pi_np1
    return pi_values, tau_values


def amplitudes(solution: MieSolution, mu_values) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu_values, dtype=float)
    if np.any(~np.isfinite(mu)) or np.any(mu < -1.0) or np.any(mu > 1.0):
        raise MieError("mu values must be finite and inside [-1,1]")
    pi_values, tau_values = angular_functions(mu, solution.n_stop)
    s1 = np.zeros_like(mu, dtype=np.complex128)
    s2 = np.zeros_like(mu, dtype=np.complex128)
    for n in range(1, solution.n_stop + 1):
        rn = float(n)
        factor = float(2 * n + 1) / (rn * float(n + 1))
        s1 += factor * (
            solution.an[n] * pi_values[n] + solution.bn[n] * tau_values[n]
        )
        s2 += factor * (
            solution.an[n] * tau_values[n] + solution.bn[n] * pi_values[n]
        )
    return s1, s2


def phase_function(solution: MieSolution, mu_values) -> np.ndarray:
                                                                                 
    mu = np.asarray(mu_values, dtype=float)
    if solution.qsca <= _SMALL:
        return np.ones_like(mu)
    s1, s2 = amplitudes(solution, mu)
    phase = 2.0 * (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (
        solution.x * solution.x * solution.qsca
    )
    phase = np.asarray(phase.real, dtype=float)
    if np.any(~np.isfinite(phase)) or np.any(phase < -1.0e-13):
        raise MieError("non-finite or materially negative S1/S2 phase")
    phase[phase < 0.0] = 0.0
    return phase
