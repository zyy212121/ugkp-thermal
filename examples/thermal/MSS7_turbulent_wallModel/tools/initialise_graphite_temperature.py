#!/usr/bin/env python3


from __future__ import annotations

import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]
CENTRES = CASE / "1" / "graphite" / "C"
TEMPERATURE = CASE / "1" / "graphite" / "T"

X_STRAIGHT = 0.0598
X_THROAT = 0.151945102963
X_EXIT = 0.2213
R_CHAMBER = 0.065
R_THROAT = 0.0118
R_EXIT = 0.0264
T_AMBIENT = 300.0
EASE_FRACTION = 0.05
DEPTHS_M = (0.0, 0.010, 0.020, 0.030, 0.040)
DEPTH_TEMPERATURES_K = (893.8356164383562, 390.0, 335.0, 300.0, 300.0)
AXIAL_PLATEAU_HALF_WIDTH_M = 0.030
AXIAL_TAPER_WIDTH_M = 0.020


def parse_cell_centres(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError(f"cannot parse internal cell centres from {path}")
    expected = int(match.group(1))
    vectors = [
        tuple(float(value) for value in values)
        for values in re.findall(
            r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",
            match.group(2),
        )
    ]
    if len(vectors) != expected:
        raise RuntimeError(f"expected {expected} cell centres, parsed {len(vectors)}")
    return vectors


def inner_radius(x: float) -> float:
    def plateau_eased_fraction(fraction: float) -> float:
        a = EASE_FRACTION
        if fraction <= a:
            u = fraction / a
            raw = a * (u**3 - 0.5 * u**4)
        elif fraction <= 1.0 - a:
            raw = 0.5 * a + fraction - a
        else:
            v = (1.0 - fraction) / a
            raw = (1.0 - a) - a * (v**3 - 0.5 * v**4)
        return raw / (1.0 - a)

    if x <= X_STRAIGHT:
        return R_CHAMBER
    if x <= X_THROAT:
        fraction = (x - X_STRAIGHT) / (X_THROAT - X_STRAIGHT)
        smooth = plateau_eased_fraction(fraction)
        return R_CHAMBER + smooth * (R_THROAT - R_CHAMBER)
    fraction = (x - X_THROAT) / (X_EXIT - X_THROAT)
    smooth = plateau_eased_fraction(fraction)
    return R_THROAT + smooth * (R_EXIT - R_THROAT)


def monotone_pchip_slopes(
    coordinates: tuple[float, ...], values: tuple[float, ...]
) -> tuple[float, ...]:
    
    if len(coordinates) != len(values) or len(coordinates) < 2:
        raise ValueError("PCHIP requires equally sized coordinate/value arrays")
    widths = [right - left for left, right in zip(coordinates, coordinates[1:])]
    if any(width <= 0.0 for width in widths):
        raise ValueError("PCHIP coordinates must be strictly increasing")
    secants = [
        (right - left) / width
        for left, right, width in zip(values, values[1:], widths)
    ]
    if len(values) == 2:
        return (secants[0], secants[0])

    slopes = [0.0] * len(values)
    for index in range(1, len(values) - 1):
        left_secant = secants[index - 1]
        right_secant = secants[index]
        if left_secant == 0.0 or right_secant == 0.0 or left_secant * right_secant < 0.0:
            slopes[index] = 0.0
            continue
        left_width = widths[index - 1]
        right_width = widths[index]
        weight_left = 2.0 * right_width + left_width
        weight_right = right_width + 2.0 * left_width
        slopes[index] = (weight_left + weight_right) / (
            weight_left / left_secant + weight_right / right_secant
        )

    def endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if slope * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(slope) > abs(3.0 * d0):
            return 3.0 * d0
        return slope

    slopes[0] = endpoint_slope(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = endpoint_slope(
        widths[-1], widths[-2], secants[-1], secants[-2]
    )
    return tuple(slopes)


DEPTH_SLOPES_K_PER_M = monotone_pchip_slopes(
    DEPTHS_M, DEPTH_TEMPERATURES_K
)


def measured_depth_temperature(depth: float) -> float:
    
    if depth <= DEPTHS_M[0]:
        return DEPTH_TEMPERATURES_K[0]
    if depth >= DEPTHS_M[-1]:
        return DEPTH_TEMPERATURES_K[-1]
    for left in range(len(DEPTHS_M) - 1):
        right = left + 1
        if depth <= DEPTHS_M[right]:
            width = DEPTHS_M[right] - DEPTHS_M[left]
            fraction = (depth - DEPTHS_M[left]) / width
            fraction2 = fraction * fraction
            fraction3 = fraction2 * fraction
            return (
                (2.0 * fraction3 - 3.0 * fraction2 + 1.0)
                * DEPTH_TEMPERATURES_K[left]
                + (fraction3 - 2.0 * fraction2 + fraction)
                * width
                * DEPTH_SLOPES_K_PER_M[left]
                + (-2.0 * fraction3 + 3.0 * fraction2)
                * DEPTH_TEMPERATURES_K[right]
                + (fraction3 - fraction2)
                * width
                * DEPTH_SLOPES_K_PER_M[right]
            )
    raise AssertionError("unreachable")


def axial_weight(x: float) -> float:
    
    distance = abs(x - X_THROAT)
    if distance <= AXIAL_PLATEAU_HALF_WIDTH_M:
        return 1.0
    if distance >= AXIAL_PLATEAU_HALF_WIDTH_M + AXIAL_TAPER_WIDTH_M:
        return 0.0
    fraction = (
        (distance - AXIAL_PLATEAU_HALF_WIDTH_M) / AXIAL_TAPER_WIDTH_M
    )
    smoothstep = fraction**3 * (
        10.0 + fraction * (-15.0 + 6.0 * fraction)
    )
    return 1.0 - smoothstep


def cell_temperature(x: float, y: float, z: float) -> float:
    radius = math.hypot(y, z)
    depth = max(0.0, radius - inner_radius(x))
    throat_profile = measured_depth_temperature(depth)
                                                                             
                                                                           
                                                                       
    return T_AMBIENT + axial_weight(x) * (throat_profile - T_AMBIENT)


def main() -> None:
    centres = parse_cell_centres(CENTRES)
    values = [cell_temperature(*centre) for centre in centres]
    source = TEMPERATURE.read_text(encoding="utf-8")
    replacement = (
        "internalField nonuniform List<scalar>\n"
        f"{len(values)}\n(\n"
        + "\n".join(f"{value:.12g}" for value in values)
        + "\n);"
    )
    updated, count = re.subn(
        r"internalField\s+(?:uniform\s+[-+0-9.eE]+|"
        r"nonuniform\s+List<scalar>\s+\d+\s*\(.*?\))\s*;",
        replacement,
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"cannot replace internalField in {TEMPERATURE}")
    TEMPERATURE.write_text(updated, encoding="utf-8")
    print(
        f"initialised {len(values)} graphite cells: "
        f"Tmin={min(values):.3f} K Tmax={max(values):.3f} K"
    )


if __name__ == "__main__":
    main()
