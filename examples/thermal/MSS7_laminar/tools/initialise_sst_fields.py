#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
from pathlib import Path


DEFAULT_U_REF = 20.14765418478137
DEFAULT_INTENSITY = 0.05
DEFAULT_LENGTH_SCALE = 0.0091
DEFAULT_C_MU = 0.09


def scalar_field(
    location: str,
    name: str,
    dimensions: str,
    internal: float,
    inlet: float,
    wall_type: str,
) -> str:
    outlet_value = inlet
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    \"{location}\";
    object      {name};
}}

dimensions      {dimensions};
internalField   uniform {internal:.17g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {inlet:.17g};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {outlet_value:.17g};
        value           uniform {outlet_value:.17g};
    }}
    fluid_to_graphite
    {{
        type            {wall_type};
        value           uniform {internal:.17g};
    }}
    wedgeBack
    {{
        type            wedge;
    }}
    wedgeFront
    {{
        type            wedge;
    }}
    defaultFaces
    {{
        type            empty;
    }}
}}
"""


def nut_field(location: str, value: float) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    \"{location}\";
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {value:.17g};

boundaryField
{{
    inlet            {{ type zeroGradient; }}
    outlet           {{ type zeroGradient; }}
    fluid_to_graphite
    {{
        type            nutUSpaldingWallFunction;
        value           uniform 0;
    }}
    wedgeBack        {{ type wedge; }}
    wedgeFront       {{ type wedge; }}
    defaultFaces     {{ type empty; }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "field_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]/"1/fluid",
    )
    parser.add_argument("--u-ref", type=float, default=DEFAULT_U_REF)
    parser.add_argument("--intensity", type=float, default=DEFAULT_INTENSITY)
    parser.add_argument("--length-scale", type=float, default=DEFAULT_LENGTH_SCALE)
    args = parser.parse_args()

    if args.u_ref <= 0 or args.intensity <= 0 or args.length_scale <= 0:
        raise SystemExit("u-ref, intensity and length-scale must be positive")
    if not args.field_dir.is_dir():
        raise SystemExit(f"field directory is absent: {args.field_dir}")

    k_value = 1.5*(args.intensity*args.u_ref)**2
    omega_value = math.sqrt(k_value)/(DEFAULT_C_MU**0.25*args.length_scale)
    nut_value = k_value/omega_value
    location = "1/fluid"

    (args.field_dir/"k").write_text(
        scalar_field(
            location,
            "k",
            "[0 2 -2 0 0 0 0]",
            k_value,
            k_value,
            "kqRWallFunction",
        ),
        encoding="utf-8",
    )
    (args.field_dir/"omega").write_text(
        scalar_field(
            location,
            "omega",
            "[0 0 -1 0 0 0 0]",
            omega_value,
            omega_value,
            "omegaWallFunction",
        ),
        encoding="utf-8",
    )
    (args.field_dir/"nut").write_text(
        nut_field(location, nut_value),
        encoding="utf-8",
    )
    print(f"fieldDir={args.field_dir.resolve()}")
    print(f"Uref={args.u_ref:.12g} intensity={args.intensity:.12g}")
    print(f"lengthScale={args.length_scale:.12g} Cmu={DEFAULT_C_MU:.12g}")
    print(f"k={k_value:.12g} omega={omega_value:.12g} nut={nut_value:.12g}")


if __name__ == "__main__":
    main()
