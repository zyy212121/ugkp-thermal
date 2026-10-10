#!/usr/bin/env python3
"""Prepare the native refinement pair; execute only when explicitly requested.

Usage: python run.py --output /path/to/new/cases
       python run.py --output /path/to/new/cases --mesh-only
       python run.py --output /path/to/new/cases --execute
Source OpenFOAM 10 first. Build the backend with UGKWP_GAS_SPECIES=10.
No native solver execution has been performed in the host-only test environment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location("native_reactor_"+name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"runs")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--mesh-only",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    status={"native_execution":"NOT_RUN","mesh":"NOT_RUN","validation":"NOT_RUN","cases":[],"commands":[],"reason":"Preparation only; use --execute on a CUDA host with the exact Ns=10 backend."}
    status_path=args.output/"run_status.json"
    def save():
        status_path.write_text(json.dumps(status,sort_keys=True,indent=2)+"\n")
    try:
        generator=load("make_case")
        for variant in ("coarse","fine"):
            case=args.output/variant
            generator.create_case(case,variant)
            status["cases"].append(str(case.resolve()))
        save()
        if args.mesh_only or args.execute:
            for executable in ("blockMesh","checkMesh"):
                if not shutil.which(executable):
                    raise RuntimeError("required OpenFOAM command is unavailable: "+executable)
            status["mesh"]="RUNNING"
            save()
            for case in map(Path,status["cases"]):
                for executable in ("blockMesh","checkMesh"):
                    command=[executable,"-case",str(case)]
                    status["commands"].append(command)
                    with (case/("log."+executable)).open("w") as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    if executable=="checkMesh" and "Mesh OK." not in (case/"log.checkMesh").read_text():
                        raise RuntimeError("checkMesh did not report Mesh OK")
            status["mesh"]="PASS"
            save()
        if args.execute:
            if not shutil.which("gasUGKP"):
                raise RuntimeError("gasUGKP unavailable; native execution was not attempted")
            for case in map(Path,status["cases"]):
                command=["gasUGKP","-case",str(case)]
                status["commands"].append(command)
                status["native_execution"]="RUNNING"
                save()
                with (case/"log.gasUGKP").open("w") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            status["native_execution"]="COMPLETED"
            result=load("check_case").compare_pair(args.output/"coarse",args.output/"fine")
            (args.output/"comparison.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
            status["validation"]="PASS" if result["passed"] else "FAIL"
            if not result["passed"]:
                raise RuntimeError("native fields failed pinned-reference/refinement/conservation checks")
            status["reason"]="Actual native output and identity passed the checker."
        elif args.mesh_only:
            status["reason"]="OpenFOAM mesh checks completed; native CUDA solver was not run."
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        status["reason"]=str(error)
        if status["mesh"]=="RUNNING":
            status["mesh"]="FAILED"
        if status["native_execution"]=="RUNNING":
            status["native_execution"]="FAILED"
        status["validation"]="FAIL" if status["native_execution"]!="NOT_RUN" else "NOT_RUN"
        save()
        print(json.dumps(status,sort_keys=True,indent=2))
        raise SystemExit(1)
    save()
    print(json.dumps(status,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
