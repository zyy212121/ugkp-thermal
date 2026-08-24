#!/usr/bin/env python3
from pathlib import Path
import re


CASE = Path(__file__).resolve().parents[2]
PRESSURE = CASE / "constant/fluid/inletPressure.table"
OUTPUT = CASE / "constant/fluid/inletVolumeFraction.table"
MOLAR_MASS = 28.970253
UNIVERSAL_GAS_CONSTANT = 8314.46261815324
INLET_TEMPERATURE = 3200.0
SOLID_DENSITY = 2800.0
SOLID_MASS_FRACTION = 0.43


def solid_volume_fraction(pressure):
    gas_constant = UNIVERSAL_GAS_CONSTANT/MOLAR_MASS
    gas_density = pressure/(gas_constant*INLET_TEMPERATURE)
    solid_specific_volume = SOLID_MASS_FRACTION/SOLID_DENSITY
    gas_specific_volume = (1.0-SOLID_MASS_FRACTION)/gas_density
    return solid_specific_volume/(solid_specific_volume+gas_specific_volume), gas_density


pairs = [
    (float(time), float(pressure))
    for time, pressure in re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s*\)", PRESSURE.read_text())
]
if not pairs:
    raise RuntimeError(f"No pressure samples found in {PRESSURE}")

lines = ["("]
for time, pressure in pairs:
    epsilon_s, _ = solid_volume_fraction(pressure)
    lines.append(f"    ({time:.9f} {epsilon_s:.17g})")
lines.append(")")
OUTPUT.write_text("\n".join(lines)+"\n")

target_time = 1.5
for index in range(len(pairs)-1):
    t0, p0 = pairs[index]
    t1, p1 = pairs[index+1]
    if t0 <= target_time <= t1:
        fraction = (target_time-t0)/(t1-t0)
        pressure = p0 + fraction*(p1-p0)
        epsilon_s, gas_density = solid_volume_fraction(pressure)
        print(f"time={target_time:.6f}")
        print(f"pressure_Pa={pressure:.12g}")
        print(f"gasDensity_kg_m3={gas_density:.12g}")
        print(f"solidMassFraction={SOLID_MASS_FRACTION:.12g}")
        print(f"solidVolumeFraction={epsilon_s:.12g}")
        break
else:
    raise RuntimeError("1.5 s is outside the pressure table")
print(f"wrote={OUTPUT}")
