#!/usr/bin/env python3
"""Create the native equal-thermo periodic gasUGKP mixture verification case."""
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("mixture_transport_reference", Path(__file__).with_name("reference.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


def header(name, kind="dictionary"):
    return f"FoamFile {{ version 2.0; format ascii; class {kind}; object {name}; }}\n"


def field(name, dimensions, value, vector=False):
    kind = "volVectorField" if vector else "volScalarField"
    if isinstance(value, list):
        internal = "nonuniform List<scalar>\n" + str(len(value)) + "\n(\n" + "\n".join(format(x, ".17g") for x in value) + "\n)"
    else:
        internal = "uniform " + (value if isinstance(value,str) else format(value,".17g"))
    return header(name, kind) + f"dimensions [{dimensions}];\ninternalField {internal};\nboundaryField\n{{\n left {{ type cyclic; }}\n right {{ type cyclic; }}\n sides {{ type empty; }}\n}}\n"


def create_case(case, cells=64, diffusivity=0.02, velocity=1.0, end_time=0.1):
    if cells < 4 or diffusivity < 0 or end_time <= 0:
        raise ValueError("at least 4 cells, nonnegative diffusion and positive end time required")
    case = Path(case)
    for directory in ("0", "constant", "system"):
        (case/directory).mkdir(parents=True, exist_ok=True)
    params = dict(cells=cells, length=1.0, velocity=velocity, diffusivity=diffusivity,
        mean=0.5, amplitude=0.2, temperature=300.0, pressure=101325.0,
        molar_mass=0.028, cp=1040.0, end_time=end_time)
    gas_r = 8.31446261815324/params["molar_mass"]
    params["rho"] = params["pressure"]/(gas_r*params["temperature"])
    params["rhoE"] = params["rho"]*((params["cp"]-gas_r)*params["temperature"]+0.5*velocity*velocity)
    (case/"case_parameters.json").write_text(json.dumps(params, sort_keys=True, indent=2)+"\n")
    dx = params["length"]/cells
    a = [_ref.species_cell_average(i*dx, (i+1)*dx, 0.0,
        **{key: params[key] for key in ("length", "velocity", "diffusivity", "mean", "amplitude")}) for i in range(cells)]
    for name,dims,value,vector in (
        ("Y_A", "0 0 0 0 0 0 0", a, False),
        ("Y_B", "0 0 0 0 0 0 0", [1-x for x in a], False),
        ("rho", "1 -3 0 0 0 0 0", params["rho"], False),
        ("p", "1 -1 -2 0 0 0 0", params["pressure"], False),
        ("T", "0 0 0 1 0 0 0", params["temperature"], False),
        ("U", "0 1 -1 0 0 0 0", f"({velocity:.17g} 0 0)", True),
        ("rhoE", "1 -1 -2 0 0 0 0", params["rhoE"], False),
        ("epsilonS", "0 0 0 0 0 0 0", 0, False),
        ("Us", "0 1 -1 0 0 0 0", "(0 0 0)", True),
        ("theta", "0 2 -2 0 0 0 0", 0, False)):
        (case/"0"/name).write_text(field(name,dims,value,vector))
    species_data = "\n".join(f""" {name}
 {{
  model linearCp;
  molarMass 0.028;
  minTemperature 100;
  maxTemperature 4000;
  coefficients (1040 0 0);
 }}""" for name in ("A","B"))
    (case/"constant/gasModelProperties").write_text(header("gasModelProperties")+f"""
schemaVersion 1;
gasMode mixtureFrozen;
species (A B);
speciesThermo
{{
{species_data}
}}
diffusion
{{
 model constant;
 coefficients ({diffusivity:.17g} {diffusivity:.17g});
 turbulentSchmidt 0.7;
}}
""")
    (case/"constant/fluidProperties").write_text(header("fluidProperties")+"""
schemaVersion 1;
thermoType { type hePsiThermo; mixture pureMixture; transport const; thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }
mixture { specie { molWeight 28; } thermodynamics { Cp 1040; Hf 0; } transport { mu 0; Pr 0.72; } }
turbulence { simulationType laminar; }
""")
    (case/"constant/particleProperties").write_text(header("particleProperties")+"""
schemaVersion 1;
gpuResidentRandomSeed 12345;
rhoS 2800;
dS 0.0002;
parcelMass 1e-6;
epsSMin 1e-12;
thetaMin 1e-12;
TgasMin 100;
""")
    (case/"constant/schedulingProperties").write_text(header("schedulingProperties")+"""
schemaVersion 1;
gpuResidentPureGasOnly true;
gpuResidentDynamicInlet false;
gpuResidentParticleCapacity 0;
gpuResidentMaxFaceWalkHops 8;
gpuResidentCourantUpdateInterval 1;
gpuResidentMaxDeltaTGrowth 1;
gpuCsrLevel L0;
gpuParticleBlockThreads 128;
gpuReductionBlockThreads 128;
""")
    (case/"system/blockMeshDict").write_text(header("blockMeshDict")+f"""
convertToMeters 1;
vertices ((0 0 0) (1 0 0) (1 0.01 0) (0 0.01 0) (0 0 0.01) (1 0 0.01) (1 0.01 0.01) (0 0.01 0.01));
blocks (hex (0 1 2 3 4 5 6 7) ({cells} 1 1) simpleGrading (1 1 1));
edges ();
boundary
(
 left {{ type cyclic; neighbourPatch right; transform translational; separationVector (-1 0 0); faces ((0 4 7 3)); }}
 right {{ type cyclic; neighbourPatch left; transform translational; separationVector (1 0 0); faces ((1 2 6 5)); }}
 sides {{ type empty; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    (case/"system/fvSchemes").write_text(header("fvSchemes")+"""
fluxScheme HLLC;
gasLimiter barthJespersen;
ddtSchemes { default SSPRK2; }
gradSchemes { default Gauss linear; }
divSchemes
{
 default none;
 div(phi,U) Gauss MUSCL;
 div(phi,e) Gauss MUSCL;
 div(phi,K) Gauss MUSCL;
 div(phi,(p|rho)) Gauss MUSCL;
 div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    (case/"system/fvSolution").write_text(header("fvSolution")+"solvers {}\nrelaxationFactors {}\nUGKP { robustFallback true; rhoMin 1e-12; TMin 100; }\n")
    (case/"system/controlDict").write_text(header("controlDict")+f"""
application gasUGKP;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end_time:.17g};
deltaT 0.00001;
writeControl runTime;
writeInterval {end_time:.17g};
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
adjustTimeStep true;
maxCo 0.3;
maxDeltaT 0.00001;
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--diffusivity", type=float, default=0.02)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--end-time", type=float, default=0.1)
    args = parser.parse_args()
    create_case(args.case,args.cells,args.diffusivity,args.velocity,args.end_time)


if __name__ == "__main__":
    main()
