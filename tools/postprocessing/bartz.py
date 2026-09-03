#!/usr/bin/env python3
from __future__ import annotations

import math


def ideal_gas_characteristic_velocity(gamma: float, gas_constant: float, total_temperature_k: float) -> float:
    return math.sqrt(gas_constant*total_temperature_k/gamma)*((gamma + 1.0)/2.0)**((gamma + 1.0)/(2.0*(gamma - 1.0)))


def bartz_throat_heat_flux(
    chamber_pressure_pa: float,
    wall_temperature_k: float,
    total_temperature_k: float,
    throat_diameter_m: float,
    throat_curvature_radius_m: float,
    dynamic_viscosity_pa_s: float,
    specific_heat_j_kg_k: float,
    prandtl: float,
    gamma: float,
    gas_constant_j_kg_k: float,
) -> tuple[float, float]:
    c_star = ideal_gas_characteristic_velocity(gamma, gas_constant_j_kg_k, total_temperature_k)
    recovery = prandtl**(1.0/3.0)
    adiabatic_wall_temperature = total_temperature_k*(1.0 + recovery*(gamma - 1.0)/2.0)/(1.0 + (gamma - 1.0)/2.0)
    temperature_ratio = wall_temperature_k/total_temperature_k
    sigma = (0.5*temperature_ratio*(1.0 + (gamma - 1.0)/2.0) + 0.5)**-0.68
    sigma *= (1.0 + (gamma - 1.0)/2.0)**-0.12
    coefficient = 0.026/(throat_diameter_m**0.2)
    coefficient *= dynamic_viscosity_pa_s**0.2*specific_heat_j_kg_k/(prandtl**0.6)
    coefficient *= (chamber_pressure_pa/c_star)**0.8
    coefficient *= (throat_diameter_m/throat_curvature_radius_m)**0.1
    heat_transfer_coefficient = coefficient*sigma
    heat_flux = heat_transfer_coefficient*(adiabatic_wall_temperature - wall_temperature_k)
    return heat_flux, heat_transfer_coefficient


def isentropic_area_ratio(mach: float, gamma: float) -> float:
    factor = (2.0/(gamma + 1.0))*(1.0 + 0.5*(gamma - 1.0)*mach*mach)
    return factor**((gamma + 1.0)/(2.0*(gamma - 1.0)))/mach


def mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool) -> float:
    ratio = max(1.0, area_ratio)
    if ratio <= 1.0 + 1.0e-12:
        return 1.0
    lower, upper = ((1.0 + 1.0e-10, 50.0) if supersonic else (1.0e-10, 1.0 - 1.0e-10))
    for _ in range(100):
        middle = 0.5*(lower + upper)
        value = isentropic_area_ratio(middle, gamma)
        if supersonic:
            if value < ratio:
                lower = middle
            else:
                upper = middle
        else:
            if value > ratio:
                lower = middle
            else:
                upper = middle
    return 0.5*(lower + upper)


def bartz_wall_heat_flux(
    chamber_pressure_pa: float,
    wall_temperature_k: float,
    local_radius_m: float,
    axial_coordinate_m: float,
    throat_axial_coordinate_m: float,
    total_temperature_k: float,
    throat_diameter_m: float,
    throat_curvature_radius_m: float,
    dynamic_viscosity_pa_s: float,
    specific_heat_j_kg_k: float,
    prandtl: float,
    gamma: float,
    gas_constant_j_kg_k: float,
) -> tuple[float, float, float]:
    throat_radius = 0.5*throat_diameter_m
    area_ratio = max(1.0, (local_radius_m/throat_radius)**2)
    mach = mach_from_area_ratio(area_ratio, gamma, axial_coordinate_m > throat_axial_coordinate_m)
    c_star = ideal_gas_characteristic_velocity(gamma, gas_constant_j_kg_k, total_temperature_k)
    recovery = prandtl**(1.0/3.0)
    temperature_factor = 1.0 + 0.5*(gamma - 1.0)*mach*mach
    adiabatic_wall_temperature = total_temperature_k*(1.0 + recovery*0.5*(gamma - 1.0)*mach*mach)/temperature_factor
    sigma = (0.5*(wall_temperature_k/total_temperature_k)*temperature_factor + 0.5)**-0.68
    sigma *= temperature_factor**-0.12
    coefficient = 0.026/(throat_diameter_m**0.2)
    coefficient *= dynamic_viscosity_pa_s**0.2*specific_heat_j_kg_k/(prandtl**0.6)
    coefficient *= (chamber_pressure_pa/c_star)**0.8
    coefficient *= (throat_diameter_m/throat_curvature_radius_m)**0.1
    coefficient *= area_ratio**-0.9
    heat_transfer_coefficient = coefficient*sigma
    heat_flux = heat_transfer_coefficient*(adiabatic_wall_temperature - wall_temperature_k)
    return heat_flux, heat_transfer_coefficient, mach
