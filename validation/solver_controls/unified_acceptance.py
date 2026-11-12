"""Unified checkpoint D extension: prepare inputs, then execute old115 plus fixed30 once.
No action occurs on import. Preparation performs file copies only, never a solver.
Short extension cases prove configuration wiring, NOT mathematical convergence.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, re, shutil, subprocess, time
HERE=Path(__file__).resolve().parent
HEADER='FoamFile { version 2.0; format ascii; class dictionary; object scheme; }\n'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def edit(path,key,value):
    # Explicit target entry edits for this fixed test only. No legacy lookup,
    # inferred aliases, include expansion or migration is performed.
    text=path.read_text();parts=key.split(".")
    start,end=0,len(text)
    for index,part in enumerate(parts[:-1]):
        match=re.search(r'(?<![\w])'+re.escape(part)+r'\s*\{',text[start:end])
        if not match:
            body=parts[-1]+" "+str(value)+";"
            for remaining in reversed(parts[index:-1]):body=remaining+" { "+body+" }"
            text=text[:end]+"\n"+body+"\n"+text[end:]
            path.write_text(text);return
        opening=start+match.end()-1;depth=1;closing=opening+1
        while depth:
            if text[closing]=='{':depth+=1
            elif text[closing]=='}':depth-=1
            closing+=1
        start,end=opening+1,closing-1
    body=text[start:end];pattern=r'(?<![\w])'+re.escape(parts[-1])+r'\s+[^;{}]+;'
    replacement=parts[-1]+" "+str(value)+";"
    if re.search(pattern,body):body=re.sub(pattern,lambda _:replacement,body,count=1)
    else:body+="\n"+replacement+"\n"
    path.write_text(text[:start]+body+text[end:])
def hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
def scheme_expected(module,key,value):
    return dict(kind="scheme",module=module,key=key,value=value,evidence="parsed typed value; reviewed static transfer chain is separate evidence")
def prepare(args):
    batch=args.batch.resolve(); source=args.source.resolve()
    dest=batch/"scheme-inputs"; dest.mkdir(exist_ok=False)
    manifest=dict(status="PREPARED_NOT_RUN",scope="interface smoke/typed transfer only",
      original_nodes=115,additional_processes=30,source=str(source),processes=[],gaps=[
      "Short cases do not prove steady or nonlinear convergence; old115 owns mathematical acceptance.",
      "Inactive chemistry, radiation, particle, three-region interface, final/distributed linear branches not runtime covered.",
      "Only CHMT has actual structure summary; other apps combine parsed value and reviewed static consumer transfer.",
      "Native old115 helpers may spawn multiple subprocesses; top-level count is not OS process count."])
    sources={"gasUGKP":"validation/shared_boundary/fixtures/gas-total","gasUGKPSteady":"validation/shared_boundary/fixtures/steady-table","CHT":"validation/molecular_transport/fixtures/cht-contact","CHMT":"validation/shared_boundary/fixtures/chmt"}
    for bits in (32,64):
        loader=batch/"fixtures"/f"loader-fp{bits}"
        for scenario in ("absent","empty","precedence","snapshot","reload"):
            ident=f"loader-{bits}-{scenario}"
            case=dest/(f"loader-{bits}-snapshot" if scenario=="reload" else ident)
            if scenario!="reload":
                (case/"constant").mkdir(parents=True);(case/"system").mkdir()
                (case/"system/controlDict").write_text(HEADER.replace("object scheme","object controlDict")+
                  "application loader; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
                if scenario!="absent":
                    shutil.copy2(HERE/"fixtures"/("empty.scheme" if scenario=="empty" else "precedence.scheme"),case/"constant/scheme")
            manifest["processes"].append(dict(id=ident,argv=[str(loader),"-case",str(case),"-scenario",scenario],
              case=str(case),depends_on=[f"loader-{bits}-snapshot"] if scenario=="reload" else [],
              expected=[dict(kind="loader")], precision=bits,app="loader"))
        for mode in ("default","override"):
            for app in ("gasUGKP","gasUGKPSteady","CHT","CHMT"):
                ident=f"{mode}-{app}-{bits}"; case=dest/ident
                shutil.copytree(source/sources[app],case)
                if app=="CHT":
                    # The supported single-gas template shares the exact existing
                    # two-region mesh definitions; copy meshes, never run blockMesh here.
                    mesh_source=source/"validation/shared_boundary/fixtures/cht"
                    for mesh in ("constant/polyMesh","constant/solid/polyMesh"):
                        shutil.copytree(mesh_source/mesh,case/mesh)
                (case/"constant/precisionProperties").write_text(HEADER.replace("object scheme","object precisionProperties")+f"precision {bits};\n")
                is_override=mode=="override"
                if not is_override:
                    # Preserve required physical scheduling while testing numerical defaults.
                    controls=HEADER
                    if app=="CHT":controls+="thermalCoupling { couplingInterval .00002; solidContactCouplingInterval .00002; radiationCouplingFrequency 1; }\n"
                    if app=="CHMT":controls+="thermalCoupling { couplingInterval .0005; }\ngasTimeStep { minDeltaT 1e-12; maxDeltaT 1e-5; maxCo .4; }\n"
                    (case/"constant/scheme").write_text(controls)
                if app=="gasUGKP":
                    if is_override:edit(case/"constant/scheme","gasTimeStep.maxCo",".4")
                    edit(case/"system/controlDict","maxCo",".123")
                    expect=[scheme_expected("gasTimeStep","maxCo",.4 if is_override else .5)]
                elif app=="gasUGKPSteady":
                    if is_override:edit(case/"constant/scheme","steadyGas.minimumIterations","2")
                    edit(case/"system/controlDict","steadyGas.minimumIterations","17")
                    expect=[scheme_expected("steadyGas","minimumIterations",2 if is_override else 1)]
                elif app=="CHT":
                    if is_override:edit(case/"constant/scheme","solid.nonlinear.nonlinearMaxIterations","57")
                    edit(case/"constant/solidRegionProperties","solidThermalCoupling.nonlinearMaxIterations","73")
                    expect=[scheme_expected("solid.nonlinear","nonlinearMaxIterations",57 if is_override else 50)]
                else:
                    if is_override:edit(case/"constant/scheme","chmt.material.conduction.nonlinearMaxIterations","57")
                    edit(case/"constant/chmtProperties","cpuMaterial.nonlinearMaxIterations","73")
                    expect=[scheme_expected("chmt.material.conduction","nonlinearMaxIterations",57 if is_override else 50),
                            dict(kind="chmt",key="materialNonlinearMaxIterations",value=57 if is_override else 50)]
                manifest["processes"].append(dict(id=ident,argv=[str(source/"bin/.precision"/str(bits)/app),"-case",str(case)],
                  case=str(case),depends_on=[],precision=bits,app=app,expected=expect,
                  physical_source=sources[app]))
        previous=None
        for stage in ("initial","restart"):
            ident=f"restart-CHMT-{bits}-{stage}"; case=dest/ident
            shutil.copytree(source/"validation/shared_boundary/fixtures/chmt",case)
            (case/"constant/precisionProperties").write_text(HEADER.replace("object scheme","object precisionProperties")+f"precision {bits};\n")
            edit(case/"system/controlDict","endTime",".0005" if stage=="initial" else ".001")
            edit(case/"constant/chmtProperties","outputDirectory",'"chmtOutput"')
            value=50 if stage=="initial" else 61
            edit(case/"constant/scheme","chmt.material.conduction.nonlinearMaxIterations",str(value))
            if previous:
                edit(case/"constant/chmtProperties","restartDirectory",'"'+str(dest/previous/"chmtOutput/checkpoint-1")+'"')
            manifest["processes"].append(dict(id=ident,argv=[str(source/"bin/.precision"/str(bits)/"CHMT"),"-case",str(case)],
               case=str(case),depends_on=[previous] if previous else [],precision=bits,app="CHMT",
               expected=[scheme_expected("chmt.material.conduction","nonlinearMaxIterations",value),
                         dict(kind="chmt",key="materialNonlinearMaxIterations",value=value)],
               checkpoint_dependency=str(dest/previous/"chmtOutput/checkpoint-1") if previous else None))
            previous=ident
    manifest["input_sha256"]=hashes(dest)
    manifest["binary_sha256"]={p["argv"][0]:sha(Path(p["argv"][0])) for p in manifest["processes"]}
    manifest["runner_sha256"]=sha(Path(__file__))
    manifest["status"]="FROZEN_NOT_RUN"
    save(batch/"scheme-manifest.json",manifest)
    print("Prepared fixed30; no solver or loader execution.")
def observe(process,text):
    findings=[]
    findings.append(dict(name="startup_once",pass_=text.count("SCHEME_STARTUP ")==1))
    for e in process["expected"]:
        if e["kind"]=="loader":
            ok=bool(re.search(r"LOADER_SUMMARY .*failures=0(?:\s|$)",text))
        elif e["kind"]=="scheme":
            rows=[dict(re.findall(r"(\w+)=([^\s]+)",line)) for line in text.splitlines() if line.startswith("SCHEME_VALUE ")]
            values=[row.get("effective") for row in rows if row.get("module")==e["module"] and row.get("key")==e["key"]]
            ok=bool(values) and all(math.isclose(float(v),e["value"],rel_tol=2e-6,abs_tol=1e-12) for v in values)
        else:
            rows=[dict(re.findall(r"(\w+)=([^\s]+)",line)) for line in text.splitlines() if line.startswith("CHMT_SOLVER_CONTROLS ")]
            ok=len(rows)==1 and math.isclose(float(rows[0].get(e["key"],"nan")),e["value"],rel_tol=2e-6)
        findings.append(dict(assertion=e,pass_=ok))
    return findings
def finite_output(case):
    checked=0; bad=[]
    numeric=re.compile(r"(?<![A-Za-z_])[-+]?(?:nan|inf|infinity)(?![A-Za-z_])",re.I)
    for directory in case.iterdir():
        if not directory.is_dir() or not (directory.name=="chmtOutput" or re.fullmatch(r"[0-9.eE+-]+",directory.name)):continue
        for p in directory.rglob("*"):
            if not p.is_file() or p.stat().st_size>32*1024*1024:continue
            if p.suffix==".csv" or p.name in ("T","U","rho","p","Tsolid","solidEnergyDensity"):
                raw=p.read_bytes()
                if b"\0" in raw:continue
                checked+=1
                if numeric.search(raw.decode(errors="replace")):bad.append(str(p))
    return dict(checked=checked,nonfinite_files=bad)
def run(args):
    batch=args.batch.resolve(); mf=batch/"scheme-manifest.json"
    plan=json.loads(mf.read_text()); dest=batch/"scheme-results";dest.mkdir(exist_ok=False)
    # Infrastructure checks apply only to this requested test runner, not production admissibility.
    for rel,h in plan["input_sha256"].items():
        if sha(batch/"scheme-inputs"/rel)!=h:raise RuntimeError("Changed frozen input "+rel)
    for path,h in plan["binary_sha256"].items():
        if sha(Path(path))!=h:raise RuntimeError("Changed frozen binary "+path)
    if sha(Path(__file__))!=plan["runner_sha256"]:raise RuntimeError("Changed frozen extension runner")
    source=Path(plan["source"]); original=batch/"batch-status.json"
    if not original.is_file() or json.loads(original.read_text()).get("status")!="FROZEN_NOT_RUN":
        raise RuntimeError("Old115 must be frozen and unstarted; refusing accidental duplicate run")
    env=dict(os.environ,OMP_NUM_THREADS="4",GAS_UGKP_CUDA_STACK_BYTES="65536",CHT_CUDA_STACK_BYTES="65536",CHMT_CUDA_STACK_BYTES="65536")
    results=dict(status="RUNNING",scope=plan["scope"],records=[],gaps=plan["gaps"])
    save(dest/"results.json",results)
    oldcmd=["bash",str(source/"validation/checkpoint03_combined/run_once.sh"),str(batch)]
    with (dest/"old115-driver.log").open("w") as out:
        old=subprocess.run(oldcmd,env=env,stdout=out,stderr=subprocess.STDOUT)
    results["original115_driver"]=dict(argv=oldcmd,exit=old.returncode)
    passed={}
    for process in plan["processes"]:
        record=dict(process);results["records"].append(record)
        missing=[x for x in process["depends_on"] if not passed.get(x)]
        if missing:
            record.update(status="SKIP_DEPENDENCY",missing=missing);passed[process["id"]]=False
            save(dest/"results.json",results);continue
        start=time.monotonic();log=dest/(process["id"]+".log")
        with log.open("w") as out:
            completed=subprocess.run(process["argv"],env=env,stdout=out,stderr=subprocess.STDOUT)
        record.update(exit=completed.returncode,seconds=time.monotonic()-start,log=str(log))
        record["observations"]=observe(process,log.read_text(errors="replace"))
        if process["app"]!="loader":record["finite_outputs"]=finite_output(Path(process["case"]))
        ok=completed.returncode==0 and all(x["pass_"] for x in record["observations"]) and not record.get("finite_outputs",{}).get("nonfinite_files",[])
        record["status"]="INTERFACE_PASS" if ok else "INTERFACE_FAIL";passed[process["id"]]=ok
        record["convergence_claim"]="NONE: short interface run only; use unchanged old115 numerical review"
        save(dest/"results.json",results)
    results["status"]="INTERFACE_BATCH_PASS_MANUAL_NUMERICAL_REVIEW_PENDING" if all(passed.values()) and old.returncode==0 else "FAIL_OR_INCOMPLETE"
    save(dest/"results.json",results)
    print(results["status"])
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("prepare");p.add_argument("--batch",type=Path,required=True);p.add_argument("--source",type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--batch",type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.mode=="prepare" else run)(args)
