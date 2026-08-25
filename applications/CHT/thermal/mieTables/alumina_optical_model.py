"""Isolated engineering optical model for high-temperature alumina."""

import math

import numpy as np


_SELLMEIER_TERMS = ((1.024, 0.00376), (1.058, 0.01225), (5.281, 321.4))
SELLMEIER_POLES_UM = tuple(math.sqrt(term[1]) for term in _SELLMEIER_TERMS)


def _n_squared_zeroes_um() -> tuple[float, ...]:
    wavelength_squared = np.poly1d([1.0, 0.0])
    denominator = np.poly1d([1.0])
    for _, pole_squared in _SELLMEIER_TERMS:
        denominator *= np.poly1d([1.0, -pole_squared])
    numerator = denominator
    for term_index, (coefficient, _) in enumerate(_SELLMEIER_TERMS):
        other_denominators = np.poly1d([1.0])
        for other_index, (_, pole_squared) in enumerate(_SELLMEIER_TERMS):
            if other_index != term_index:
                other_denominators *= np.poly1d([1.0, -pole_squared])
        numerator += coefficient * wavelength_squared * other_denominators
    zeroes = []
    for root in np.roots(numerator):
        if abs(root.imag) <= 1.0e-10 * max(abs(root.real), 1.0) and root.real > 0.0:
            zeroes.append(math.sqrt(float(root.real)))
    return tuple(sorted(zeroes))


N_SQUARED_ZEROES_UM = _n_squared_zeroes_um()


def _legacy_n_squared(wavelength_m: float) -> float:
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength must be finite and positive")
    wavelength_um = wavelength_m * 1.0e6
    lambda_squared = wavelength_um * wavelength_um
    for _, pole_squared in _SELLMEIER_TERMS:
        denominator = lambda_squared - pole_squared
        if abs(denominator) <= 64.0 * np.finfo(float).eps * max(lambda_squared, pole_squared):
            raise ValueError(
                f"legacy alumina Sellmeier pole at {math.sqrt(pole_squared):.17g} um"
            )
    value = 1.0 + lambda_squared * sum(
        coefficient / (lambda_squared - pole_squared)
        for coefficient, pole_squared in _SELLMEIER_TERMS
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"legacy alumina nSquared must be finite and >0 at {wavelength_um:.17g} um; got {value}"
        )
    return value


def validate_band(wavelength_min_m: float, wavelength_max_m: float) -> None:
    """Reject every legacy band crossing a pole or a nonpositive-n² interval."""
    if not (
        math.isfinite(wavelength_min_m)
        and math.isfinite(wavelength_max_m)
        and 0.0 < wavelength_min_m < wavelength_max_m
    ):
        raise ValueError("legacy alumina band requires finite 0 < min < max")
    minimum_um = wavelength_min_m * 1.0e6
    maximum_um = wavelength_max_m * 1.0e6
    for pole_um in SELLMEIER_POLES_UM:
        if minimum_um <= pole_um <= maximum_um:
            raise ValueError(
                f"legacy alumina band contains Sellmeier pole at {pole_um:.17g} um"
            )
    for zero_um in N_SQUARED_ZEROES_UM:
        if minimum_um <= zero_um <= maximum_um:
            raise ValueError(
                f"legacy alumina band crosses nSquared=0 at {zero_um:.17g} um"
            )
    _legacy_n_squared(wavelength_min_m)
    _legacy_n_squared(wavelength_max_m)
    _legacy_n_squared(0.5 * (wavelength_min_m + wavelength_max_m))


def refractive_index(wavelength_m: float, temperature_k: float) -> complex:
    """Return the legacy engineering alumina refractive index ``n - i*k``."""
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be finite and positive")

    wavelength_um = wavelength_m * 1.0e6
    lambda_squared = wavelength_um * wavelength_um
    n_base = math.sqrt(_legacy_n_squared(wavelength_m))
    n_real = n_base * (0.9904 + 2.02e-5 * temperature_k)
    try:
        k_absorption = (
            0.002
            * (0.06 * lambda_squared + 0.7 * wavelength_um + 1.0)
            * math.exp(1.847 * (temperature_k / 1000.0 - 2.95))
        )
    except OverflowError as error:
        raise ValueError("legacy alumina absorption index is nonfinite") from error
    if not math.isfinite(n_real) or n_real <= 0.0:
        raise ValueError("legacy alumina refractive index n must be finite and positive")
    if not math.isfinite(k_absorption) or k_absorption < 0.0:
        raise ValueError("legacy alumina absorption index k must be finite and nonnegative")
    return complex(n_real, -k_absorption)
