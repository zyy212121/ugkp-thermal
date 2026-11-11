#!/usr/bin/env python3
"""Freeze locally built inputs, then explicitly run one ordered acceptance batch."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, platform, shutil, subprocess, threading, time, traceback
from acceptance_support import failed_environment_scope
ACCEPTED="a3793e59dbfd6a9120ca80e4bb653c771c90ee59"
SOLVERS=("gasUGKP","gasUGKPSteady","CHT","CHMT")
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def capture(argv,cwd=None):
    try:
        p=subprocess.run(argv,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        return dict(command=list(map(str,argv)),exit=p.returncode,output=p.stdout)
    except OSError as error:
        return dict(command=list(map(str,argv)),exit=None,output=repr(error))
def freeze(args):
    root=args.root.resolve(); fixture=args.fixtures.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False); source=out/"source"; source.mkdir()
    selected=subprocess.check_output(["git","-C",str(root),"ls-files","-z","--cached","--others","--exclude-standard"])
    names=set(os.fsdecode(n) for n in selected.split(b"\0") if n)
    prefixes={"applications","common","gpu","legacy","tool","third_party","validation","precision"}
    rows={}; paths=[]
    for name in sorted(names):
        rel=Path(name); p=root/rel
        if rel.parts[0] not in prefixes and name not in ("Allwmake","env.sh",".gitignore"): continue
        if not p.is_file() and not p.is_symlink(): continue
        # No README is created or modified by this acceptance package.
        if p.name.lower().startswith("readme"): continue
        dst=source/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if p.is_symlink(): dst.symlink_to(os.readlink(p))
        else: shutil.copy2(p,dst)
        rows[name]=dict(sha256=digest(p),symlink=os.readlink(p) if p.is_symlink() else None)
        paths.append(p)
    shutil.copytree(fixture,out/"fixtures")
    for bits in ("32","64"):
        for solver in SOLVERS:
            src=root/"bin/.precision"/bits/solver
            dst=source/"bin/.precision"/bits/solver; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
    for solver in SOLVERS:
        dst=source/"bin"/solver; shutil.copy2(root/"bin"/solver,dst)
    materialized={}
    for bits in ("32","64"):
        file=root/".build/precision"/("fp"+bits)/"source-manifest.json"
        data=json.loads(file.read_text()); current={}
        tree=file.parent/"source"
        for name in data["files"]:
            p=tree/name
            current[name]=digest(p) if p.is_file() else None
        materialized[bits]={"manifest":data,"actual_sha256":current}
    compiled=json.loads((out/"fixtures/fixtures.json").read_text())
    command_files=[]
    for item in compiled["tasks"]:
        command_files.append(out/"fixtures"/item["executable"])
    # Freeze original FP64 evidence and enumerate every root change for review.
    original=capture(["git","diff",ACCEPTED,"--","common","gpu","applications"],root)
    (out/"fp64-root-vs-checkpoint01.patch").write_text(original["output"])
    base_tree=subprocess.check_output(["git","-C",str(root),"ls-tree","-r","-z",ACCEPTED,"--","common","gpu","applications"])
    original_objects={}
    for entry in base_tree.split(b"\0"):
        if not entry:continue
        info,name=entry.split(b"\t",1)
        if Path(os.fsdecode(name)).name.lower().startswith("readme"):continue
        original_objects[os.fsdecode(name)]=info.split()[2].decode()
    current_objects={}
    for name in rows:
        if Path(name).parts[0] not in {"common","gpu","applications"}:continue
        file=root/name
        content=os.fsencode(os.readlink(file)) if file.is_symlink() else file.read_bytes()
        current_objects[name]=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
    comparison={"accepted_base":ACCEPTED,"changed":[],"added":[],"missing":[]}
    for name,oid in current_objects.items():
        if name not in original_objects:comparison["added"].append(name)
        elif oid!=original_objects[name]:comparison["changed"].append(name)
    comparison["missing"]=sorted(set(original_objects)-set(current_objects))
    save(out/"fp64-base-comparison.json",comparison)
    env_keys=("WM_PROJECT_DIR","WM_OPTIONS","WM_PRECISION_OPTION","FOAM_LIBBIN","LD_LIBRARY_PATH",
              "CUDA_HOME","CUDA_ARCH","OMP_NUM_THREADS","CUDA_VISIBLE_DEVICES")
    metadata={"status":"FROZEN_NOT_RUN","created":now(),"accepted_base":ACCEPTED,
        "candidate_root":str(root),"head":capture(["git","rev-parse","HEAD"],root),
        "git_status":capture(["git","status","--short"],root),
        "git_diff":capture(["git","diff","--binary","HEAD"],root),
        "source_files":rows,"materialized":materialized,"environment":{k:os.environ.get(k) for k in env_keys},
        "host":platform.platform(),"gpu":capture(["nvidia-smi"]),"fixtures":compiled,
        "readme_note":"README files intentionally not copied or changed.",
        "limits":["Checkpoint02 old results are not new pass evidence.",
                  "FP64 root diff needs human review: only intended precision IO/build glue may differ.",
                  "This freeze records current binaries. Matching source-to-binary build logs remain required."]}
    save(out/"freeze.json",metadata)
    hashes={}
    for top in (source,out/"fixtures"):
        for p in sorted(top.rglob("*")):
            if p.is_file(): hashes[str(p.relative_to(out))]=digest(p)
    save(out/"frozen-sha256.json",hashes)
    save(out/"batch-status.json",{"status":"FROZEN_NOT_RUN","created":now()})
    print(out)
def sample_gpu(stop,path):
    with path.open("w") as stream:
        stream.write("utc,index,memory_used_MiB,gpu_utilization_percent,power_W,sm_clock_MHz\n")
        while not stop.is_set():
            try:
                p=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
                for line in (p.stdout if p.returncode==0 else "UNAVAILABLE,"+p.stderr.strip()).splitlines():
                    stream.write(now()+","+line+"\n")
                stream.flush()
            except Exception as error: stream.write(now()+",UNAVAILABLE,"+repr(error)+"\n");stream.flush()
            stop.wait(.2)
def run(args):
    out=args.batch.resolve(); frozen=json.loads((out/"freeze.json").read_text())
    results=out/"results"; results.mkdir(exist_ok=False)
    status={"status":"RUNNING","started":now(),"steps":[],"not_run":[],"failures":[],"reused":[]}
    save(out/"batch-status.json",status)
    try:
        for rel,expected in json.loads((out/"frozen-sha256.json").read_text()).items():
            if digest(out/rel)!=expected: raise RuntimeError("Frozen input changed: "+rel)
        source=out/"source"; fixture=out/"fixtures"
        env=dict(os.environ,UGKP_SOURCE_ROOT=str(source),UGKP_ACCEPTANCE_BIN=str(source/"bin"),
                 PYTHONDONTWRITEBYTECODE="1")
        reuse=json.loads((fixture/"reuse-pass.json").read_text())
        tasks=[]
        cpu_names={"central-host","double_geometry_intersection","double_wall_source","mixed_chemistry",
                   "mixed_eos_rk","mixed_divergence_ale","mixed_particle_energy-0","mixed_particle_energy-1",
                   "mixed_coupling_energy-1","model-regressions","material-wall"}
        for item in frozen["fixtures"]["tasks"]:
            tasks.append(dict(id=item["id"],argv=[str(fixture/item["executable"])]+item["args"],
                precision=item["precision"],coverage=item["coverage"],executable=item["executable"],
                dependencies=[],gpu=Path(item["executable"]).name not in cpu_names))
        adapter=source/"validation/checkpoint02/acceptance/native_dual.py"
        for bits in (64,32):
            tasks.append(dict(id="native-fp"+str(bits),precision=bits,gpu=True,dependencies=[],
                subreport=str(results/("native-fp"+str(bits))/"results.json"),
                argv=["python3",str(adapter),"--precision",str(bits),"--output",str(results/("native-fp"+str(bits))),
                      "--probe",str(fixture/frozen["fixtures"]["probe"][str(bits)])],
                coverage="Four solvers; original 17 native branches per precision, dependency-aware collection."))
        tasks.append(dict(id="restart-matrix",precision="both",gpu=True,dependencies=[],
            subreport=str(results/"restart-matrix/results.json"),
            argv=["python3",str(source/"validation/checkpoint02/acceptance/restart_matrix.py"),
                "--output",str(results/"restart-matrix"),"--fixtures",str(fixture)],
            coverage="Original 44 restart branches with valid producer dependencies."))
        for repeat in range(3):
            for bits in ((64,32) if repeat%2==0 else (32,64)):
                required=["fp%d-radiation-pair-%d"%(bits,i) for i in range(3)]
                required+=(["fp64-radiation-original-%d"%i for i in range(3)] if bits==64 else
                           ["fp32-ordered-radiation","fp32-radiation-acceptance"])
                tasks.append(dict(id="performance-radiation-"+str(repeat)+"-fp"+str(bits),precision=bits,
                    gpu=True,dependencies=required,
                    argv=[str(fixture/("fp"+str(bits))/"radiation-pair"),"1024","8","chain"],
                    coverage="Matched repeated cold-start workload; corresponding radiation correctness required."))
        passed={};environment_failure=None
        for index,item in enumerate(tasks):
            stepdir=results/("step-%03d-%s"%(index,item["id"]));stepdir.mkdir()
            record=dict(item,started=now());status["steps"].append(record)
            if item["id"] in reuse["tasks"]:
                prior=reuse["tasks"][item["id"]]
                try:
                    normalized=[]
                    for arg in item["argv"]:
                        normalized.append(arg.replace(str(source),"$SOURCE").replace(str(fixture),"$FIXTURES").replace(str(results),"$RESULTS"))
                    if prior["original_result"] not in ("RUN_PASS","REUSED_PASS") or prior["normalized_argv"]!=normalized:
                        raise RuntimeError("Prior success/input mismatch")
                    if prior["original_result"]=="RUN_PASS" and prior["previous_execution"].get("exit")!=0:
                        raise RuntimeError("Prior execution did not succeed")
                    for relative,expected in prior["launch_files"].items():
                        if digest(out/relative)!=expected:raise RuntimeError("Reused program changed: "+relative)
                    for relative,expected in prior["evidence"].items():
                        if digest(fixture/relative)!=expected:raise RuntimeError("Reused evidence changed: "+relative)
                    for bits in prior["dependency_precisions"]:
                        for relative,expected in reuse["dependency_hashes"][bits].items():
                            if frozen["materialized"][bits]["actual_sha256"].get(relative)!=expected:
                                raise RuntimeError("Reused source dependency changed: "+relative)
                    record.update(result="REUSED_PASS",reused_evidence=prior,original_result=prior["original_result"],
                                  original_batch=prior["source_batch"],original_wall_seconds=prior["previous_execution"].get("wall_seconds"))
                    for name in ("stdout.log","gpu.csv"):
                        if name in prior.get("copied_step_files",{}):
                            shutil.copy2(fixture/prior["copied_step_files"][name],stepdir/name)
                    if prior["previous_execution"].get("branches"):
                        record["branches"]=[dict(branch,result="REUSED_PASS",original_result=branch["result"],
                            original_batch=prior["source_batch"]) for branch in prior["previous_execution"]["branches"]]
                    status["reused"].append(item["id"])
                except Exception as error:
                    record.update(result="FAIL",error=repr(error),execution="NOT_EXECUTED_INVALID_REUSE")
            else:
                missing=[key for key in item["dependencies"] if not passed.get(key)]
                invalid=environment_failure and (environment_failure["scope"]=="all" or item["gpu"])
                if missing or invalid:
                    record.update(result="SKIP_DEPENDENCY",reason=environment_failure if invalid else {"failed_dependencies":missing})
                    status["not_run"].append(item["id"])
                else:
                    save(out/"batch-status.json",status)
                    stop=threading.Event();monitor=threading.Thread(target=sample_gpu,args=(stop,stepdir/"gpu.csv"),daemon=True);monitor.start()
                    start=time.perf_counter()
                    try:
                        with (stepdir/"stdout.log").open("w") as log:
                            proc=subprocess.run(item["argv"],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                        record["exit"]=proc.returncode
                        record["result"]="RUN_PASS" if proc.returncode==0 else "FAIL"
                    except Exception as error:
                        record.update(result="FAIL",error=repr(error))
                    finally:
                        stop.set();monitor.join(timeout=7);record["wall_seconds"]=time.perf_counter()-start
                    if item.get("subreport") and Path(item["subreport"]).is_file():
                        child=json.loads(Path(item["subreport"]).read_text());record["branches"]=child.get("branches",[])
                        if record["result"]=="FAIL" and child.get("environment_failure"):
                            environment_failure=child["environment_failure"]
                    if record["result"]=="FAIL" and not environment_failure:
                        text=(stepdir/"stdout.log").read_text(errors="replace") if (stepdir/"stdout.log").exists() else record.get("error","")
                        environment_failure=failed_environment_scope(text,record.get("exit"),item["gpu"])
            record["finished"]=now();passed[item["id"]]=record["result"] in ("RUN_PASS","REUSED_PASS")
            save(stepdir/"execution.json",record)
            if record["result"]=="FAIL":
                status["failures"].append(item["id"])
                save(stepdir/"failure-evidence.sha256.json",{p.name:digest(p) for p in stepdir.iterdir() if p.is_file()})
                if not (out/"first-failure.json").exists():save(out/"first-failure.json",record)
            if environment_failure:status["environment_failure"]=environment_failure
            save(out/"batch-status.json",status)
        status["status"]="PASS_EXECUTION_REVIEW_REQUIRED" if all(passed.values()) else "FAIL_COLLECTED_REVIEW_REQUIRED"
        status["limits"]=["Every required item must have valid RUN_PASS or REUSED_PASS coverage; failures and dependency skips are never passes.",
            "GPU samples are device-level observed memory, not exact allocation peaks or attributable per-process VRAM.",
            "WSL may report N/A; no missing telemetry is represented as zero.",
            "All paired solves use solve(false); compare matched repeated cold-start timings, not warm starts.",
            "Outer source_metric=0 is not proof of FP32 physical residual 1e-12.",
            "Short native cases do not establish production steady-state convergence."]
    except BaseException as error:
        status["status"]="FAIL_STOPPED";status["error"]=repr(error)
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["finished"]=now();save(out/"batch-status.json",status)
        save(out/"evidence-sha256.json",{str(p.relative_to(out)):digest(p) for p in results.rglob("*") if p.is_file()})
        summary=out/"source/validation/checkpoint02/acceptance/summarize.py"
        if summary.exists():
            with (out/"summary.log").open("w") as log:
                subprocess.run(["python3",str(summary),str(out)],stdout=log,stderr=subprocess.STDOUT)
    if status["status"]!="PASS_EXECUTION_REVIEW_REQUIRED":raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest="action",required=True)
    f=subs.add_parser("freeze");f.add_argument("--root",type=Path,required=True)
    f.add_argument("--fixtures",type=Path,required=True);f.add_argument("--output",type=Path,required=True)
    r=subs.add_parser("run");r.add_argument("--batch",type=Path,required=True)
    a=p.parse_args();freeze(a) if a.action=="freeze" else run(a)
if __name__=="__main__":main()
