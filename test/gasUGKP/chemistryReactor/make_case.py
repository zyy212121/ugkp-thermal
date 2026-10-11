#!/usr/bin/env python3
"""Generate a native uniform periodic full-H2/O2/N2 chemistry case.

Reuses the mixtureTransport case and field helpers. Generation is host-only;
it is not evidence that the native CUDA solver has run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MECHANISMS = ROOT/"common/chemistry/mechanisms"
REFERENCE = MECHANISMS/"h2o2.cantera-reactor-reference.json"
REFERENCE_SHA256 = "e2a640c6396dd0694aadf741e1f6aef21a819dd550d6e9a8a0743f0d347886b6"
CASE_ID = "nitrogen_1200K_1atm"

_spec = importlib.util.spec_from_file_location("native_mixture_case_helpers", HERE.parent/"mixtureTransport/make_case.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def reference_case():
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("pinned Cantera reactor reference SHA-256 changed")
    return next(case for case in json.loads(REFERENCE.read_text())["cases"] if case["name"] == CASE_ID)


def create_case(case, variant="fine"):
    if variant not in ("coarse", "fine"):
        raise ValueError("variant must be coarse or fine")
    case = Path(case)
    for entry in case.iterdir() if case.exists() else []:
        try:
            output_time = float(entry.name)
        except ValueError:
            continue
        if entry.is_dir() and output_time > 0:
            raise ValueError("refusing to regenerate a case containing solver time directories")
    reference = reference_case()
    initial = reference["states"][0]
    manifest = json.loads((MECHANISMS/"h2o2.manifest.json").read_text())
    species = manifest["species"]
    controls = {"relativeTolerance": 1e-4 if variant == "coarse" else 1e-7,
                "absoluteMassFractionTolerance": 1e-14, "absoluteTemperatureTolerance": 1e-6,
                "maximumSteps": 100000, "maximumRejectedSteps": 1000}
    dt = 1e-6 if variant == "coarse" else 5e-7
    _base.create_case(case, cells=4, diffusivity=0, velocity=0, end_time=reference["actual_horizon"])
    # These two files were just created by the reused passive-mixture helper.
    (case/"0/Y_A").unlink()
    (case/"0/Y_B").unlink()
    for name, value in zip(species, initial["massFractions"]):
        (case/"0"/("Y_"+name)).write_text(_base.field("Y_"+name,"0 0 0 0 0 0 0",value))
    for name, dimensions, value, vector in (
        ("rho","1 -3 0 0 0 0 0",initial["density"],False),
        ("rhoU","1 -2 -1 0 0 0 0","(0 0 0)",True),
        ("rhoE","1 -1 -2 0 0 0 0",initial["internalEnergy"]/reference["volume"],False),
        ("U","0 1 -1 0 0 0 0","(0 0 0)",True),
        ("p","1 -1 -2 0 0 0 0",initial["pressure"],False),
        ("T","0 0 0 1 0 0 0",initial["temperature"],False)):
        (case/"0"/name).write_text(_base.field(name,dimensions,value,vector))
    gas_model = (MECHANISMS/"h2o2.gasModelProperties").read_text()
    gas_model += "\nchemistryControls\n{\n"+"".join("    "+key+" "+format(value,".17g")+";\n" for key,value in controls.items())+"}\n"
    (case/"constant/gasModelProperties").write_text(_base.header("gasModelProperties")+gas_model)
    shutil.copyfile(MECHANISMS/"h2o2.mechanism",case/"constant/h2o2.mechanism")
    (case/"system/blockMeshDict").write_text(_base.header("blockMeshDict")+"""
convertToMeters 1;
vertices ((0 0 0) (0.01 0 0) (0.01 0.01 0) (0 0.01 0) (0 0 0.01) (0.01 0 0.01) (0.01 0.01 0.01) (0 0.01 0.01));
blocks (hex (0 1 2 3 4 5 6 7) (4 1 1) simpleGrading (1 1 1));
edges ();
boundary
(
 left { type cyclic; neighbourPatch right; transform translational; separationVector (-0.01 0 0); faces ((0 4 7 3)); }
 right { type cyclic; neighbourPatch left; transform translational; separationVector (0.01 0 0); faces ((1 2 6 5)); }
 sides { type empty; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
""")
    schemes = (case/"system/fvSchemes").read_text().replace("HLLC", "Tadmor").replace("barthJespersen", "none").replace("SSPRK2", "Euler").replace("Gauss MUSCL", "Gauss upwind")
    (case/"system/fvSchemes").write_text(schemes)
    (case/"system/controlDict").write_text(_base.header("controlDict")+f"""
application gasUGKP;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {reference['actual_horizon']:.17g};
deltaT {dt:.17g};
writeControl runTime;
writeInterval 1e-6;
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
adjustTimeStep false;
maxCo 0.3;
maxDeltaT {dt:.17g};
""")
    contract = {
        "schema_version": 1, "application": "gasUGKP", "native_execution": "NOT_RUN",
        "variant": variant, "cells": 4, "volume": 1e-6, "cell_volume": 2.5e-7,
        "reference_case": CASE_ID, "reference_path": str(REFERENCE.relative_to(ROOT)), "reference_sha256": REFERENCE_SHA256,
        "species": species, "reaction_count": 29, "phase": "ohmech",
        "initial_temperature": 1200, "initial_pressure": 101325,
        "initial_mole_ratios": reference["composition"], "initial_inventories": initial,
        "end_time": reference["actual_horizon"], "delta_t": dt, "write_interval": 1e-6,
        "check_times": [s["time"] for s in reference["states"] if s["time"] == 0 or s["time"] >= 1e-6-1e-15],
        "chemistry_controls": controls,
        "required_build": {"api": 1,"UGKWP_GAS_SPECIES":10,"mode":2,
                           "speciesOrderHash":manifest["species_order_hash"],"thermoHash":manifest["thermo_hash"],"mechanismHash":manifest["mechanism_hash"]},
        "required_fields": ["rho","rhoU","rhoE","U","p","T"]+["Y_"+name for name in species],
        "native_command": ["gasUGKP","-case","CASE_DIRECTORY"],
        "build_command": "UGKWP_GAS_SPECIES=10 applications/gasUGKP/private_backend/build_private_backend.sh",
        "numerics": {"fluxScheme":"Tadmor","reconstruction":"firstOrder","timeIntegrator":"Euler","sourceComposition":"C(dt/2) -> transport(dt) -> C(dt/2)"},
        "acceptance": {
            "provenance": "tests/chemistry/test_h2o2_reactor.py:test_chemical_tolerance_refinement_reduces_history_error; native path remains unvalidated until executed",
            "history_metric": "max over output times and cells of abs(T-Tref)/3000 K and all abs(Y-Yref)",
            "temperature_scale_K":3000,"fine_max_normalized_history_error":1e-5,
            "fine_to_coarse_error_ratio":0.15,
            "reference_uncertainty_floor": max(reference["comparison"]["max_abs_temperature_difference"]/3000,reference["comparison"]["max_abs_mass_fraction_difference"]),
            "conservation_relative":2e-9,"species_sum_absolute":2e-10,"nonnegative_mass_fraction_absolute":1e-14,
            "energy_closure_relative":2e-8,"zero_velocity_absolute_m_per_s":1e-10,
            "conservation_basis": "Closed periodic uniform cell; fixed rho and rhoE, zero momentum, element moles conserved. Energy reconstructed from full NASA7 formation-inclusive e(T,Y)."
        }
    }
    (case/"case_contract.json").write_text(json.dumps(contract,sort_keys=True,indent=2)+"\n")
    (case/"case_parameters.json").write_text(json.dumps({"fixture":"chemistryReactor","contract":"case_contract.json"},sort_keys=True,indent=2)+"\n")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,default=HERE/"case")
    parser.add_argument("--variant",choices=("coarse","fine"),default="fine")
    args=parser.parse_args()
    contract=create_case(args.case,args.variant)
    print(json.dumps({"case":str(args.case),"reference_case":contract["reference_case"],"native_execution":"NOT_RUN"},indent=2))


if __name__ == "__main__":
    main()
