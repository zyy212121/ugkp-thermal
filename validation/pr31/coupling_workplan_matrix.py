#!/usr/bin/env python3
"""Serial production CHMT matrix. Fresh synthetic cases only; never edits baselines.
Reports actual inventories, output differences, warnings, and process termination.
Compact is covered separately by validation/tests/run_flat_pressure_cuda.py,
which directly executes the production CUDA kernels, not a full solver variant.
"""
import argparse, ast, json, math, re, subprocess, time
from pathlib import Path
import chmt_particles_native as fixture

def replace_block(text, key, body):
    match=re.search(r"\b"+re.escape(key)+r"\s*\{",text)
    if not match: raise ValueError("missing patch "+key)
    start=text.index("{",match.start());end=start+1;depth=1
    while depth:
        depth+=(text[end]=="{")-(text[end]=="}");end+=1
    return text[:start+1]+body+text[end-1:]

def prepare(case, level, injection):
    fixture.prepare(case,"coldWall1D",False,end_time=2e-6)
    schedule=case/"constant/schedulingProperties"
    for key,value in [("gpuCsrLevel",level),("gpuCsrHeavyReductionAutoInterval",1),
                      ("gpuResidentParticleCapacity",4096)]:
        fixture.set_input(schedule,key,value)
    fixture.set_input(case/"constant/scheme","couplingInterval",1e-6)
    fixture.set_input(case/"constant/scheme","maxDeltaT",1e-7)
    boundary=(case/"constant/polyMesh/boundary").read_text()
    patch=re.search(r"\bleft\s*\{([^}]+)",boundary)[1]
    face=int(re.search(r"startFace\s+(\d+)",patch)[1])
    # 512 mobile + one attached in cell 0; cell 1 has only one mobile.
    # The original 2x2x2 native fixture fixes these cell centres.
    rows=[]
    for i in range(513):
        cell=0 if i<512 else 1
        x=.25 if cell==0 else .75
        rows.append(f'p{i} {{id "{i+1}"; rng "{1234+i}"; cell {cell}; '
                    f'position ({x} .25 .25); velocity (.01 0 0); '
                    'mass 1e-10; diameter .0001; temperature 2500;}')
    cp=lambda t:(153.5+.0031*(t-2327.))/.101961276
    h=1273.*2317.+(.5*(1273.+cp(2327.))+1.16e6/20.)*20.+.5*(cp(2337.)+cp(2500.))*163.
    nodes=" ".join(str(h+i) for i in range(8))
    rows.append(f'attached {{id "1000"; rng "987654"; cell 0; face {face}; state 2; '
                'position (0 .25 .25); velocity (0 0 0); mass 1e-10; diameter .0001; '
                'temperature 2500; contactAge .0001; contactDuration .001; contactArea 0; '
                'thermalHistory {temperature 2500; savedVelocity (0 1 0); '
                'contactMaximumArea 1e-8; contactPeakFraction .4; '
                f'coldNodeSpecificEnthalpy ({nodes}); coldRingSolidMass (0 0 0 0 0 0 0 0);'
                '}}')
    (case/"constant/particleProperties").write_text(
        "FoamFile {version 2.0; format ascii; class dictionary; object particleProperties;}\n"
        "rhoS 2500; dS .0001; parcelMass 1e-10; volumeMode Coupled;\n"
        "dragModel SchillerNaumann; particleGasHeatTransferModel RanzMarshall;\n"
        "collisionalPressure true; gpuResidentInjectionTp 2500;\n"
        "gpuResidentStuckModel {wallInteractionModel coldWall1D; heatTransfer true; "
        "interfaceThermalResistance .00001;}\nparticles {\n"+"\n".join(rows)+"\n}\n")
    if injection:
        u=case/"0/U"
        u.write_text(replace_block(u.read_text(),"right","type fixedValue; value uniform (-.1 0 0);"))
        (case/"0/epsilonS").write_text(
            "FoamFile {version 2.0; format ascii; class volScalarField; object epsilonS;}\n"
            "dimensions [0 0 0 0 0 0 0]; internalField uniform 0;\n"
            "boundaryField {left {type zeroGradient;} sides {type zeroGradient;} "
            "right {type fixedValue; value uniform 1e-8;}}\n")

def snapshot(case):
    out=case/"chmtOutput"
    indices=sorted(int(p.stem.split("-")[1]) for p in out.glob("particles-*.csv"))
    if len(indices)<2: raise RuntimeError("no accepted particle evolution output")
    first,last=fixture.inventory(out,indices[0]),fixture.inventory(out,indices[-1])
    for row in last["parcels"]:
        for key,value in row.items():
            if key!="id" and not math.isfinite(float(value)):
                raise RuntimeError(f"nonfinite particle field {key}")
    result={"first":first,"last":last,"accepted_output_indices":indices}
    result["energy_change_J"]=sum(last[k]-first[k] for k in ("gas","solid","particle"))
    result["mass_change_kg"]=sum(last[k]-first[k] for k in ("gas_mass","particle_mass"))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--work",type=Path,required=True)
    ap.add_argument("--fp32",type=Path,required=True);ap.add_argument("--fp64",type=Path,required=True)
    ap.add_argument("--prepare-only",action="store_true")
    args=ap.parse_args();root=args.work.resolve();root.mkdir(parents=True,exist_ok=False)
    reports=[];references={}
    variants=[("ordinary",32,args.fp32),("ordinary",64,args.fp64)]
    for variant,bits,exe in variants:
        for level in ("L0","L1","L2","auto"):
            for injection in (False,True):
                name=f"{variant}-fp{bits}-{level}-"+("injection" if injection else "closed")
                case=root/name
                report={"name":name,"precision":bits,"level":level,"variant":variant,
                        "initial_cell_particle_counts":[513,1,0,0,0,0,0,0],
                        "injection_requested":injection,
                        "directory_expectation":"full on initial step; split after committed directory",
                        "compact_requested":variant=="compact"}
                reports.append(report)
                try:
                    prepare(case,level,injection)
                    if args.prepare_only:report["status"]="prepared";continue
                    start=time.perf_counter()
                    with (case/"run.log").open("w") as log:
                        run=subprocess.run([str(exe.resolve()),"-case",str(case)],
                            stdout=log,stderr=subprocess.STDOUT,env=fixture.m.ENV)
                    report["elapsed_seconds"]=time.perf_counter()-start
                    report["returncode"]=run.returncode
                    text=(case/"run.log").read_text(errors="replace")
                    report["warnings"]=[line for line in text.splitlines() if re.search(r"warn|fatal|budget|terminate",line,re.I)]
                    if run.returncode:raise RuntimeError("solver terminated; inspect run.log")
                    report.update(snapshot(case));report["status"]="completed"
                    key=(variant,bits,injection)
                    if level=="L0":references[key]=report["last"]
                    if key in references:
                        base=references[key]
                        report["difference_from_L0"]={k:report["last"][k]-base[k] for k in
                            ("gas","solid","particle","gas_mass","particle_mass")}
                except Exception as error:
                    report["status"]="failed";report["error"]=str(error)
                finally:
                    (root/"results.json").write_text(json.dumps(reports,indent=2)+"\n")
    coverage={"launch_trace_verified":False,
              "compact_native_reachable":False,
              "compact_test_command":"python3 validation/tests/run_flat_pressure_cuda.py --work <fresh-directory>",
              "notes":["Directory expectations are not a CUDA trace proof.",
                       "Compare energy/mass changes with open-boundary fluxes; not zero-residual assertions.",
                       "This does not replace the long CHT restart macro baseline.",
                       "Auto mode is exercised but its selected heavy-list decision requires diagnostics."]}
    (root/"coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
    if any(r["status"]=="failed" for r in reports):raise SystemExit(1)
if __name__=="__main__":main()
