#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path


CASE = Path(__file__).resolve().parents[2]
ASSET = Path(__file__).resolve().parent
PRECASE = ASSET / "preconditioning_case"
EXPERIMENT = CASE / "assets/experimental/graphite_thermocouples.csv"


def foam_header(location: str, name: str, cls: str) -> str:
    return f'''FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    location "{location}";
    object {name};
}}
'''


def scalar_field(location: str, name: str, dimensions: str, value: float,
                 inlet: str, outlet: str, wall: str, coupled: str) -> str:
    return foam_header(location, name, "volScalarField") + f'''
dimensions {dimensions};
internalField uniform {value:.17g};
boundaryField
{{
    inlet {{ {inlet} }}
    outlet {{ {outlet} }}
    symmetry {{ type symmetry; }}
    focusWall {{ {wall} }}
    walls {{ {wall} }}
    fluid_to_graphite {{ {coupled} }}
}}
'''


def vector_field(location: str, name: str, dimensions: str,
                 value: tuple[float, float, float], inlet: str,
                 outlet: str, wall: str, coupled: str) -> str:
    v = f"({value[0]:.17g} {value[1]:.17g} {value[2]:.17g})"
    return foam_header(location, name, "volVectorField") + f'''
dimensions {dimensions};
internalField uniform {v};
boundaryField
{{
    inlet {{ {inlet} }}
    outlet {{ {outlet} }}
    symmetry {{ type symmetry; }}
    focusWall {{ {wall} }}
    walls {{ {wall} }}
    fluid_to_graphite {{ {coupled} }}
}}
'''


def write_pre_fields(root: Path) -> None:
    field = root / "1.4"
    field.mkdir(parents=True, exist_ok=True)
    pin = 6030241.0
    ambient_p = 101325.0
    ambient_t = 300.0
    (field / "p").write_text(scalar_field(
        "1.4", "p", "[1 -1 -2 0 0 0 0]", ambient_p,
        f"type fixedValue; value uniform {pin};",
        "type waveTransmissive; psi psi; gamma 1.4; fieldInf 101325; lInf 0.004; value uniform 101325;",
        "type zeroGradient;", "type zeroGradient;"))
    (field / "T").write_text(scalar_field(
        "1.4", "T", "[0 0 0 1 0 0 0]", ambient_t,
        "type fixedValue; value uniform 3200;", "type zeroGradient;",
        "type fixedValue; value uniform 480;",
        "type fixedValue; value uniform 480;"))
    (field / "U").write_text(vector_field(
        "1.4", "U", "[0 1 -1 0 0 0 0]", (0, 0, 0),
        "type zeroGradient;",
        "type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);",
        "type noSlip;", "type noSlip;"))
    (field / "Tp").write_text(scalar_field(
        "1.4", "Tp", "[0 0 0 1 0 0 0]", 3200,
        "type fixedValue; value uniform 3200;", "type zeroGradient;",
        "type zeroGradient;", "type zeroGradient;"))
    (field / "epsilonS").write_text(scalar_field(
        "1.4", "epsilonS", "[0 0 0 0 0 0 0]", 0,
        "type fixedValue; value uniform 0;", "type zeroGradient;",
        "type zeroGradient;", "type zeroGradient;"))
    (field / "theta").write_text(scalar_field(
        "1.4", "theta", "[0 2 -2 0 0 0 0]", 0,
        "type fixedValue; value uniform 0;", "type zeroGradient;",
        "type zeroGradient;", "type zeroGradient;"))
    (field / "Us").write_text(vector_field(
        "1.4", "Us", "[0 1 -1 0 0 0 0]", (0, 0, 0),
        "type fixedValue; value uniform (0 0 0);", "type zeroGradient;",
        "type zeroGradient;", "type zeroGradient;"))
    k = 1.52223
    omega = 247.536
    (field / "k").write_text(scalar_field(
        "1.4", "k", "[0 2 -2 0 0 0 0]", k,
        f"type fixedValue; value uniform {k};",
        f"type inletOutlet; inletValue uniform {k}; value uniform {k};",
        "type kqRWallFunction; value uniform 0;",
        "type kqRWallFunction; value uniform 0;"))
    (field / "omega").write_text(scalar_field(
        "1.4", "omega", "[0 0 -1 0 0 0 0]", omega,
        f"type fixedValue; value uniform {omega};",
        f"type inletOutlet; inletValue uniform {omega}; value uniform {omega};",
        "type omegaWallFunction; beta1 0.075; blended 0; value uniform 247.536;",
        "type omegaWallFunction; beta1 0.075; blended 0; value uniform 247.536;"))
    (field / "nut").write_text(scalar_field(
        "1.4", "nut", "[0 2 -1 0 0 0 0]", 0,
        "type zeroGradient;", "type zeroGradient;",
        "type zeroGradient;", "type zeroGradient;"))


def create_preconditioning_case() -> None:
    if PRECASE.exists():
        shutil.rmtree(PRECASE)
    (PRECASE / "constant").mkdir(parents=True)
    (PRECASE / "system").mkdir()
    shutil.copytree(CASE / "constant/fluid/polyMesh", PRECASE / "constant/polyMesh")
    for name in ("fluidProperties", "inletPressure.table", "inletVolumeFraction.table", "particleProperties"):
        shutil.copy2(CASE / "constant/fluid" / name if name in {"fluidProperties", "inletPressure.table", "inletVolumeFraction.table"} else CASE / "constant" / name, PRECASE / "constant" / name)
    particle = (PRECASE / "constant/particleProperties").read_text()
    particle = re.sub(r"^gpuResidentSolidThermalCoupling\s+true;\s*$", "", particle, flags=re.M)
    (PRECASE / "constant/particleProperties").write_text(particle)
    scheduling = (CASE / "constant/fluid/schedulingProperties").read_text()
    scheduling = scheduling.replace("gpuResidentPureGasOnly          false;", "gpuResidentPureGasOnly          true;")
    (PRECASE / "constant/schedulingProperties").write_text(scheduling)
    (PRECASE / "constant/radiationProperties").write_text(foam_header("constant", "radiationProperties", "dictionary") + "\nschemaVersion 1;\nenabled false;\nmodel {}\n")
    (PRECASE / "constant/solidRegionProperties").write_text(foam_header("constant", "solidRegionProperties", "dictionary") + "\nschemaVersion 1;\nenabled false;\n")
    shutil.copy2(CASE / "system/fluid/fvSchemes", PRECASE / "system/fvSchemes")
    shutil.copy2(CASE / "system/fluid/fvSolution", PRECASE / "system/fvSolution")
    (PRECASE / "system/controlDict").write_text(foam_header("system", "controlDict", "dictionary") + '''
application gasUGKP;
startFrom startTime;
startTime 1.4;
stopAt endTime;
endTime 1.5;
deltaT 1e-8;
writeControl runTime;
writeInterval 0.1;
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
writeCompression off;
timeFormat general;
timePrecision 17;
runTimeModifiable true;
adjustTimeStep yes;
maxCo 0.5;
maxDeltaT 1e-5;
''')
    write_pre_fields(PRECASE)
    print(PRECASE)


def parse_internal(path: Path, kind: str):
    text = path.read_text()
    if kind == "scalar":
        uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
        if uniform:
            return float(uniform.group(1)), None
        match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
        return None, [float(x) for x in match.group(2).split()]
    uniform = re.search(r"internalField\s+uniform\s+\(([^)]+)\)\s*;", text)
    if uniform:
        return tuple(map(float, uniform.group(1).split())), None
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    return None, [tuple(map(float, x.split())) for x in re.findall(r"\(([^)]+)\)", match.group(2))]


def values(parsed, count: int):
    uniform, nonuniform = parsed
    return [uniform] * count if nonuniform is None else nonuniform


def nonuniform_scalar(location: str, name: str, dimensions: str, data, boundary: str) -> str:
    body = "\n".join(f"{x:.17g}" for x in data)
    return foam_header(location, name, "volScalarField") + f'''\ndimensions {dimensions};\ninternalField nonuniform List<scalar>\n{len(data)}\n(\n{body}\n);\nboundaryField\n{{\n{boundary}\n}}\n'''


def nonuniform_vector(location: str, name: str, dimensions: str, data, boundary: str) -> str:
    body = "\n".join(f"({x[0]:.17g} {x[1]:.17g} {x[2]:.17g})" for x in data)
    return foam_header(location, name, "volVectorField") + f'''\ndimensions {dimensions};\ninternalField nonuniform List<vector>\n{len(data)}\n(\n{body}\n);\nboundaryField\n{{\n{boundary}\n}}\n'''


def formal_scalar_boundary(inlet: str, outlet: str, wall: str, coupled: str) -> str:
    return f'''    inlet {{ {inlet} }}\n    outlet {{ {outlet} }}\n    symmetry {{ type symmetry; }}\n    focusWall {{ {wall} }}\n    walls {{ {wall} }}\n    fluid_to_graphite {{ {coupled} }}'''


def install_formal_state(source_time: str) -> None:
    source = PRECASE / source_time
    if not source.is_dir():
        raise SystemExit(f"missing preconditioning result {source}")
    target = CASE / "1.5"
    if target.exists():
        shutil.rmtree(target)
    fluid = target / "fluid"
    graphite = target / "graphite"
    fluid.mkdir(parents=True)
    graphite.mkdir()
    for name in ("p", "U", "T", "k", "omega", "nut"):
        text = (source / name).read_text().replace(f'location    "{source_time}";', 'location    "1.5/fluid";').replace(f'location "{source_time}";', 'location "1.5/fluid";')
        if name == "T":
            text = re.sub(r"(focusWall\s*\{.*?value\s+uniform\s+)[-+0-9.eE]+", r"\g<1>750", text, flags=re.S)
            text = re.sub(r"(walls\s*\{.*?value\s+uniform\s+)[-+0-9.eE]+", r"\g<1>750", text, flags=re.S)
        (fluid / name).write_text(text)
    p = values(parse_internal(source / "p", "scalar"), 5304)
    u = values(parse_internal(source / "U", "vector"), len(p))
    t = values(parse_internal(source / "T", "scalar"), len(p))
    rho = [pp / (287.0 * tt) for pp, tt in zip(p, t)]
    rhou = [(rr*vv[0], rr*vv[1], rr*vv[2]) for rr, vv in zip(rho, u)]
    rhoe = [pp/0.4 + 0.5*rr*sum(q*q for q in vv) for pp, rr, vv in zip(p, rho, u)]
    inlet_pressure = 6238838.0
    rho_bc = formal_scalar_boundary(f"type fixedValue; value uniform {inlet_pressure/(287.0*3200.0):.17g};", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;")
    energy_bc = formal_scalar_boundary(f"type fixedValue; value uniform {inlet_pressure/0.4:.17g};", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;")
    vector_bc = formal_scalar_boundary("type zeroGradient;", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;")
    (fluid / "rho").write_text(nonuniform_scalar("1.5/fluid", "rho", "[1 -3 0 0 0 0 0]", rho, rho_bc))
    (fluid / "rhoU").write_text(nonuniform_vector("1.5/fluid", "rhoU", "[1 -2 -1 0 0 0 0]", rhou, vector_bc))
    (fluid / "rhoE").write_text(nonuniform_scalar("1.5/fluid", "rhoE", "[1 -1 -2 0 0 0 0]", rhoe, energy_bc))
    zero_scalar = {
        "epsilonS": "[0 0 0 0 0 0 0]", "rhoEs": "[1 -1 -2 0 0 0 0]",
        "rhoDs": "[1 -2 0 0 0 0 0]", "rhoHp": "[1 -1 -2 0 0 0 0]",
        "theta": "[0 2 -2 0 0 0 0]", "dMeanCell": "[0 1 0 0 0 0 0]"
    }
    zero_bc = formal_scalar_boundary("type fixedValue; value uniform 0;", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;")
    for name, dims in zero_scalar.items():
        (fluid / name).write_text(scalar_field("1.5/fluid", name, dims, 0, "type fixedValue; value uniform 0;", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;"))
    for name in ("rhoUs", "Us"):
        dims = "[1 -2 -1 0 0 0 0]" if name == "rhoUs" else "[0 1 -1 0 0 0 0]"
        (fluid / name).write_text(vector_field("1.5/fluid", name, dims, (0, 0, 0), "type fixedValue; value uniform (0 0 0);", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;"))
    (fluid / "Tp").write_text(scalar_field("1.5/fluid", "Tp", "[0 0 0 1 0 0 0]", 3200, "type fixedValue; value uniform 3200;", "type zeroGradient;", "type zeroGradient;", "type zeroGradient;"))
    for name in ("particleStuckWallHeatFlux", "particleReflectedWallHeatFlux"):
        (fluid / name).write_text(foam_header("1.5/fluid", name, "surfaceScalarField") + "\ndimensions [1 0 -3 0 0 0 0];\ninternalField uniform 0;\nboundaryField { inlet {type calculated; value uniform 0;} outlet {type calculated; value uniform 0;} symmetry {type symmetry;} focusWall {type calculated; value uniform 0;} walls {type calculated; value uniform 0;} fluid_to_graphite {type calculated; value uniform 0;} }\n")
    shutil.copy2(CASE / "2.500000008807/graphite/T", graphite / "T")
    text = (graphite / "T").read_text().replace('location    "2.500000008807/graphite";', 'location    "1.5/graphite";').replace("internalField   uniform 750;", "internalField   uniform 350;")
    text = text.replace("value           uniform 750;", "value           uniform 493.71;", 1)
    text = text.replace("value           uniform 750;", "value           uniform 299.187256;", 1)
    (graphite / "T").write_text(text)
    (target / "gpuResidentStrictEpsGPrev.dat").write_text(
        f"{len(p)}\n" + "\n".join("1.00000000000000000e+00" for _ in p) + "\n"
    )
    boundary = (CASE / "constant/fluid/polyMesh/boundary").read_text()
    inlet = re.search(r"inlet\s*\{.*?nFaces\s+(\d+);.*?startFace\s+(\d+);", boundary, re.S)
    if not inlet:
        raise RuntimeError("cannot resolve inlet patch face range")
    count, start = map(int, inlet.groups())
    residual = [f"UGKP_SOURCE_RESIDUAL_SCHEMA1 {count}"]
    residual.extend(f"{start+i} 0.00000000000000000e+00" for i in range(count))
    (target / "gpuResidentStrictSourceResidual.dat").write_text("\n".join(residual) + "\n")
    shutil.rmtree(CASE / "2.500000008807")
    print(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "install"))
    parser.add_argument("--source-time", default="1.5")
    args = parser.parse_args()
    if args.action == "create":
        create_preconditioning_case()
    else:
        install_formal_state(args.source_time)


if __name__ == "__main__":
    main()
