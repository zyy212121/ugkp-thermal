#!/usr/bin/env python3
"""Validate real native fields against pinned chemistry and conservation evidence.

This reads native OpenFOAM ASCII fields; it does not run or replace a solver.
NASA7 energy/pressure reconstruction is independent of the native C++ calorics.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location("reactor_case_definition",HERE/"make_case.py")
_generator=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator)
_spec=importlib.util.spec_from_file_location("pinned_chemistry_import",_generator.MECHANISMS/"import_h2o2.py")
_importer=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_importer)
_MODEL=_importer.load_mechanism(_generator.MECHANISMS/_importer.SOURCE_NAME)
R=8.31446261815324


def thermodynamics(temperature, mass_fractions):
    if not 300 <= temperature <= 3500:
        raise ValueError("temperature outside common pinned NASA7 validity interval")
    gas_constant=0.0
    energy=0.0
    elements=[0.0]*4
    for y,sp in zip(mass_fractions,_MODEL["thermo"]):
        a=sp["coefficients"][:7] if temperature<=sp["midTemperature"] else sp["coefficients"][7:]
        t=temperature
        h_over_rt=a[0]+a[1]*t/2+a[2]*t*t/3+a[3]*t**3/4+a[4]*t**4/5+a[5]/t
        energy+=y*R*t*(h_over_rt-1)/sp["molarMass"]
        gas_constant+=y*R/sp["molarMass"]
        for e,atoms in enumerate(sp["atoms"]):
            elements[e]+=y*atoms/sp["molarMass"]
    return {"specific_internal_energy":energy,"gas_constant":gas_constant,"element_moles_per_kg":elements}


def read_field(path, count, vector=False):
    path=Path(path)
    text=path.read_text()
    uniform=re.search(r"internalField\s+uniform\s+([^;]+);",text)
    if uniform:
        raw=uniform.group(1).strip()
        value=[float(x) for x in raw.strip("()").split()] if vector else float(raw)
        if vector and len(value)!=3:
            raise ValueError("invalid vector field "+str(path))
        values=[value]*count
    else:
        kind="vector" if vector else "scalar"
        match=re.search(r"internalField\s+nonuniform\s+List<"+kind+r">\s+(\d+)\s*\((.*?)\)\s*;",text,re.S)
        if not match or int(match[1])!=count:
            raise ValueError("wrong or absent ASCII internalField "+str(path))
        values=[[float(x) for x in row.split()] for row in re.findall(r"\(([^()]*)\)",match[2])] if vector else [float(x) for x in match[2].split()]
        if len(values)!=count or (vector and any(len(row)!=3 for row in values)):
            raise ValueError("wrong number of field entries "+str(path))
    flat=[v for row in values for v in row] if vector else values
    if not all(math.isfinite(v) for v in flat):
        raise ValueError("nonfinite field "+str(path))
    return values


def check_identity(case, contract):
    log=(case/"log.gasUGKP").read_text()
    matches=re.findall(r"Shared gas model:\s*api=(\d+)\s+Ns=(\d+)\s+mode=(\d+)\s+speciesOrderHash=(\d+)\s+thermoHash=(\d+)\s+mechanismHash=(\d+)",log)
    identity=contract["required_build"]
    expected=tuple(str(identity[k]) for k in ("api","UGKWP_GAS_SPECIES","mode","speciesOrderHash","thermoHash","mechanismHash"))
    if not matches or any(match!=expected for match in matches):
        raise ValueError("missing or mismatched native model/build identity")
    if not re.search(r"^End\s*$",log,re.M) or re.search(r"FOAM FATAL|CUDA error|chemistry.*(?:unsupported|failed)",log,re.I):
        raise ValueError("native solver log does not establish successful completion")


def time_directory(case,target):
    candidates=[]
    for path in case.iterdir():
        try:
            value=float(path.name)
        except ValueError:
            continue
        if path.is_dir() and abs(value-target)<=max(1e-15,abs(target)*1e-10):
            candidates.append(path)
    if len(candidates)!=1:
        raise ValueError("missing or ambiguous native output at time "+str(target))
    return candidates[0]


def compare_case(case):
    case=Path(case)
    result={"case":str(case),"passed":False,"failures":[],"normalized_history_error":0.0,"checked_times":[],
            "max_relative_density_error":0.0,"max_relative_energy_inventory_error":0.0,
            "max_relative_energy_closure_error":0.0,"max_relative_pressure_closure_error":0.0,
            "max_species_sum_error":0.0,"max_element_relative_error":0.0,"max_velocity":0.0}
    try:
        contract=json.loads((case/"case_contract.json").read_text())
        if contract["reference_sha256"]!=_generator.REFERENCE_SHA256 or contract["reference_case"]!=_generator.CASE_ID:
            raise ValueError("case declares a different reference identity")
        reference=_generator.reference_case()
        expected_times=[state["time"] for state in reference["states"] if state["time"]==0 or state["time"]>=1e-6-1e-15]
        if contract["check_times"]!=expected_times or contract["cells"]!=4 or contract["volume"]!=1e-6:
            raise ValueError("case changed the required complete history, cell count or volume")
        expected_identity={"api":1,"UGKWP_GAS_SPECIES":10,"mode":2,
                           "speciesOrderHash":str(_MODEL["speciesOrderHash"]),
                           "thermoHash":str(_MODEL["thermoHash"]),
                           "mechanismHash":str(_MODEL["mechanismHash"])}
        if contract["required_build"]!=expected_identity or contract["species"]!=_MODEL["species"]:
            raise ValueError("case changed pinned species order or model identity")
        config=(case/"constant/gasModelProperties").read_text()
        controls=contract["chemistry_controls"]
        expected_tolerance={"coarse":1e-4,"fine":1e-7}.get(contract["variant"])
        if controls["relativeTolerance"]!=expected_tolerance:
            raise ValueError("case changed the verified refinement controls")
        for key,value in controls.items():
            found=re.findall(r"\b"+re.escape(key)+r"\s+([^;]+);",config)
            if len(found)!=1 or float(found[0])!=value:
                raise ValueError("actual chemistryControls differs from case contract: "+key)
        check_identity(case,contract)
        result["variant"]=contract["variant"]
        n=contract["cells"]
        accept=contract["acceptance"]
        initial=reference["states"][0]
        rho0=initial["density"]
        energy0=initial["internalEnergy"]/contract["volume"]
        atom0=[v/n for v in initial["elementMoles"]]
        failures=result["failures"]
        for target in contract["check_times"]:
            reference_state=next(state for state in reference["states"] if abs(state["time"]-target)<1e-15)
            directory=time_directory(case,target)
            fields={name:read_field(directory/name,n,name in ("U","rhoU")) for name in contract["required_fields"]}
            for cell in range(n):
                y=[fields["Y_"+name][cell] for name in contract["species"]]
                temperature=fields["T"][cell]
                density=fields["rho"][cell]
                if density<=0 or min(y)<-accept["nonnegative_mass_fraction_absolute"]:
                    failures.append("nonpositive density or negative species at "+str(target))
                total_y=abs(sum(y)-1)
                result["max_species_sum_error"]=max(result["max_species_sum_error"],total_y)
                thermo=thermodynamics(temperature,y)
                velocity=fields["U"][cell]
                momentum=fields["rhoU"][cell]
                speed=max(abs(v) for v in velocity)
                result["max_velocity"]=max(result["max_velocity"],speed,max(abs(v)/rho0 for v in momentum))
                if any(abs(m-rho0*u)>rho0*accept["zero_velocity_absolute_m_per_s"] for m,u in zip(momentum,velocity)):
                    failures.append("rhoU does not equal rho*U at "+str(target))
                e_reconstructed=density*(thermo["specific_internal_energy"]+sum(v*v for v in velocity)/2)
                p_reconstructed=density*thermo["gas_constant"]*temperature
                result["max_relative_density_error"]=max(result["max_relative_density_error"],abs(density-rho0)/rho0)
                result["max_relative_energy_inventory_error"]=max(result["max_relative_energy_inventory_error"],abs(fields["rhoE"][cell]-energy0)/abs(energy0))
                result["max_relative_energy_closure_error"]=max(result["max_relative_energy_closure_error"],abs(fields["rhoE"][cell]-e_reconstructed)/max(abs(energy0),abs(e_reconstructed)))
                result["max_relative_pressure_closure_error"]=max(result["max_relative_pressure_closure_error"],abs(fields["p"][cell]-p_reconstructed)/p_reconstructed)
                for atoms,expected in zip(thermo["element_moles_per_kg"],atom0):
                    observed=atoms*density*contract["cell_volume"]
                    result["max_element_relative_error"]=max(result["max_element_relative_error"],abs(observed-expected)/max(abs(expected),1e-20))
                result["normalized_history_error"]=max(result["normalized_history_error"],abs(temperature-reference_state["temperature"])/accept["temperature_scale_K"],max(abs(value-expected) for value,expected in zip(y,reference_state["massFractions"])))
            result["checked_times"].append(target)
        for metric,tolerance in (("max_relative_density_error",accept["conservation_relative"]),
                                 ("max_relative_energy_inventory_error",accept["conservation_relative"]),
                                 ("max_relative_energy_closure_error",accept["energy_closure_relative"]),
                                 ("max_relative_pressure_closure_error",accept["energy_closure_relative"]),
                                 ("max_element_relative_error",accept["conservation_relative"]),
                                 ("max_species_sum_error",accept["species_sum_absolute"]),
                                 ("max_velocity",accept["zero_velocity_absolute_m_per_s"])):
            if result[metric]>tolerance:
                failures.append(metric+" exceeds "+str(tolerance))
        if contract["variant"]=="fine" and result["normalized_history_error"]>accept["fine_max_normalized_history_error"]:
            failures.append("fine history exceeds the verified common-core error envelope")
        result["passed"]=not failures
    except (OSError,ValueError,KeyError,StopIteration) as error:
        result["failures"].append(str(error))
    return result


def compare_pair(coarse,fine):
    coarse_result,fine_result=compare_case(coarse),compare_case(fine)
    contract=json.loads((Path(fine)/"case_contract.json").read_text())
    tolerance=contract["acceptance"]
    threshold=max(tolerance["fine_to_coarse_error_ratio"]*coarse_result["normalized_history_error"],4*tolerance["reference_uncertainty_floor"])
    roles=coarse_result.get("variant")=="coarse" and fine_result.get("variant")=="fine"
    refinement=roles and fine_result["normalized_history_error"]<=threshold
    return {"passed":coarse_result["passed"] and fine_result["passed"] and refinement,
            "coarse":coarse_result,"fine":fine_result,"refinement_passed":refinement,"refinement_threshold":threshold,
            "refinement_meaning":"Chemical tolerance and outer time step tightened together; a successful result validates this uniform native path only."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path)
    parser.add_argument("--coarse",type=Path)
    parser.add_argument("--fine",type=Path)
    args=parser.parse_args()
    if args.coarse and args.fine:
        result=compare_pair(args.coarse,args.fine)
    elif args.case:
        result=compare_case(args.case)
    else:
        parser.error("supply --case or both --coarse and --fine")
    print(json.dumps(result,sort_keys=True,indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()
