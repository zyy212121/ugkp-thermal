#!/usr/bin/env python3
"""Checkpoint03: freeze explicitly built inputs, then one declared, non-retrying batch."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, os, platform, shutil, statistics
import subprocess, threading, time, traceback
from acceptance_support import failed_environment_scope
BASE = "8bde9d6d0bf4638ad498b803a5c26f921dab2ae7"
FAMILIES = ("pressure", "face", "coupling", "bath")
VARIANTS = ("baseline", "candidate")
BRANCHES = ("gasUGKP-continuous", "gasUGKPSteady-continuous", "CHT-continuous",
            "CHMT-continuous", "hot-constant", "hot-nonlinear", "closed-drag-heat",
            "gas-wave", "gas-ode", "cht-contact", "chmt-diffusion")
SOLVERS = ("gasUGKP", "gasUGKPSteady", "CHT", "CHMT")
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def capture(argv,cwd=None):
    p=subprocess.run(list(map(str,argv)),cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(argv=list(map(str,argv)),exit=p.returncode,output=p.stdout)
def tracked(root):
    raw=subprocess.check_output(["git","-C",str(root),"ls-files","-z","--cached","--others","--exclude-standard"])
    return sorted(set(os.fsdecode(x) for x in raw.split(b"\0") if x))
def freeze(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiled=args.compiled.resolve();build=json.loads((compiled/"compiled.json").read_text())
    assert build["status"]=="COMPILED_NOT_RUN", "Compile preparation is incomplete"
    shutil.copytree(compiled,out/"fixtures")
    roots={"baseline":args.baseline.resolve(),"candidate":args.candidate.resolve()}
    controls=Path(__file__).resolve().parent
    metadata={"status":"FROZEN_NOT_RUN","created":now(),"accepted_baseline":BASE,
              "roots":{},"build":build,"host":platform.platform(),"gpu":capture(["nvidia-smi"]),
              "environment":{k:os.environ.get(k) for k in ("WM_PROJECT_DIR","WM_OPTIONS","FOAM_LIBBIN",
                 "LD_LIBRARY_PATH","CUDA_HOME","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS")}}
    for key,program in build["programs"].items():
        assert digest(compiled/program["executable"])==program["sha256"],"Program changed since compile: "+key
        assert digest(compiled/program["performance_executable"])==program["performance_sha256"],"Performance binary changed: "+key
        for name,expected in program["fixture_sources"].items():
            if (controls/name).is_file():
                assert digest(controls/name)==expected,"Fixture source changed since compile: "+name
    for variant,root in roots.items():
        dest=out/"source"/variant; dest.mkdir(parents=True)
        head=capture(["git","rev-parse","HEAD"],root)
        if variant=="baseline":assert head["output"].strip()==BASE,head
        for name in tracked(root):
            rel=Path(name)
            if rel.parts[0] not in {"common","applications","validation","tool","third_party","gpu","legacy","precision"} and name not in {"env.sh","Allwmake",".gitignore"}:continue
            if rel.name.lower().startswith("readme"):continue
            src=root/rel
            if not src.is_file() and not src.is_symlink():continue
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if src.is_symlink():target.symlink_to(os.readlink(src))
            else:shutil.copy2(src,target)
        # New acceptance code is identical for both variants. Production remains distinct.
        target=dest/"validation/checkpoint03";target.mkdir(exist_ok=True)
        for src in controls.iterdir():
            if src.is_file() and not src.name.lower().startswith("readme"):shutil.copy2(src,target/src.name)
        for solver in SOLVERS:
            for rel in (Path("bin")/solver,Path("bin/.precision/32")/solver,Path("bin/.precision/64")/solver):
                dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dst)
        sources={}
        for bits in (32,64):
            info=build["variants"][variant][str(bits)]
            materialized=Path(info["tree"])
            actual={name:digest(materialized/name) for name in info["source_sha256"]}
            assert actual==info["source_sha256"], "Materialized source changed since compile"
            sources[str(bits)]={"tree":str(materialized),"sha256":actual,"manifest":info.get("manifest")}
        metadata["roots"][variant]={"original":str(root),"head":head,
             "status":capture(["git","status","--short"],root),
             "diff":capture(["git","diff","--binary","HEAD"],root),"materialized":sources,
             "backend_sha256":{f"{bits}/{s}":digest(root/"bin/.precision"/str(bits)/s)
                               for bits in (32,64) for s in SOLVERS}}
    save(out/"freeze.json",metadata)
    hashes={str(p.relative_to(out)):digest(p) for top in (out/"source",out/"fixtures")
            for p in sorted(top.rglob("*")) if p.is_file()}
    save(out/"frozen-sha256.json",hashes)
    save(out/"batch-status.json",{"status":"FROZEN_NOT_RUN","created":now()})
    save(out/"matrix.json",make_matrix(out,build))
    save(out/"matrix-sha256.json",{"matrix.json":digest(out/"matrix.json")})
    print(out)
def make_matrix(out,build):
    nodes=[]
    for bits in (64,32):
        for variant in VARIANTS:
            for family in FAMILIES:
                p=build["programs"][f"{variant}-fp{bits}-{family}"]
                nodes.append(dict(id=f"{variant}-fp{bits}-{family}",kind="correctness",variant=variant,
                    precision=bits,family=family,gpu=p["gpu"],dependencies=[],
                    argv=[str(out/"fixtures"/p["executable"])]+p.get("correctness_args",["correctness"])))
            for branch in BRANCHES:
                root=out/"source"/variant
                p=build["programs"][f"{variant}-fp{bits}-probe"]
                nodes.append(dict(id=f"{variant}-fp{bits}-native-{branch}",kind="native",variant=variant,
                    precision=bits,branch=branch,gpu=True,dependencies=[],
                    argv=["python3",str(root/"validation/checkpoint03/native.py"),"--precision",str(bits),
                          "--probe",str(out/"fixtures"/p["executable"]),"--branch",branch,
                          "--output",str(out/"results"/f"native-{variant}-fp{bits}-{branch}")] ))
        for family in FAMILIES:
            nodes.append(dict(id=f"compare-fp{bits}-{family}",kind="comparison",precision=bits,
                 family=family,gpu=False,dependencies=[f"{v}-fp{bits}-{family}" for v in VARIANTS]))
    for repeat in range(3):
        for bits in (64,32):
            for family in FAMILIES:
                for variant in (VARIANTS if repeat%2==0 else tuple(reversed(VARIANTS))):
                    p=build["programs"][f"{variant}-fp{bits}-{family}"]
                    nodes.append(dict(id=f"perf-{repeat}-{variant}-fp{bits}-{family}",kind="performance",
                      variant=variant,precision=bits,family=family,repeat=repeat,gpu=p["gpu"],
                      dependencies=[f"compare-fp{bits}-{family}"],
                      argv=[str(out/"fixtures"/p.get("performance_executable",p["executable"]))]+p.get("performance_args",["perf"])))
    return {"status":"FROZEN_NOT_RUN","node_count":len(nodes),"correctness":16,"native":44,
            "comparisons":8,"performance":48,"nodes":nodes,"new_runs_only":True,
            "seed_policy":"Host fixtures deterministic. Native gas/CHT seed12345 explicit; CHMT configuration default1 retained. Input hashes recorded before launch.",
            "order":"Correctness before performance. Paired performance AB/BA/AB within same precision/family."}
def gpu_samples(stop,path):
    with path.open("w") as stream:
        stream.write("utc,index,memory_used_MiB,gpu_utilization_percent,power_W,sm_clock_MHz,temperature_C\n")
        while not stop.is_set():
            try:
                r=capture(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu",
                           "--format=csv,noheader,nounits"])
                for line in r["output"].splitlines():stream.write(now()+","+line+"\n")
                stream.flush()
            except Exception as e:stream.write(now()+",UNAVAILABLE,"+repr(e)+"\n");stream.flush()
            stop.wait(.2)
def payload(path):
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("RESULT_JSON="):return json.loads(line.split("=",1)[1])
        if line.strip().startswith("{"):
            try:return json.loads(line)
            except ValueError:pass
    raise ValueError("No final fixture JSON record")
def numbers(data,prefix=""):
    result={}
    if isinstance(data,dict):
        for key,value in data.items():result.update(numbers(value,prefix+"/"+str(key)))
    elif isinstance(data,list):
        for key,value in enumerate(data):result.update(numbers(value,prefix+"/"+str(key)))
    elif isinstance(data,(int,float)) and not isinstance(data,bool):result[prefix]=float(data)
    else:raise ValueError("Measurements must be numeric: "+prefix)
    return result
def compare(node,records):
    left,right=(records[key]["payload"] for key in node["dependencies"])
    a=numbers(left["measurements"]);b=numbers(right["measurements"])
    assert a and a.keys()==b.keys(),"Nonmatching physical observation sets"
    rt,at=(5e-4,5e-5) if node["precision"]==32 else (1e-9,1e-10)
    errors={}
    for key in a:
        error=abs(a[key]-b[key]);allowed=at+rt*max(abs(a[key]),abs(b[key]))
        errors[key]={"baseline":a[key],"candidate":b[key],"error":error,"allowed":allowed}
        assert math.isfinite(a[key]) and math.isfinite(b[key]),(key,errors[key])
        # These are independently bounded diagnostic errors, not solution states.
        diagnostic=("residual" in key.lower() or "budget_error" in key.lower())
        errors[key]["diagnostic_only"]=diagnostic
        if not diagnostic:assert error<=allowed,(key,errors[key])
    return dict(relative_tolerance=rt,absolute_tolerance=at,measurements=errors)
def run(args):
    out=args.batch.resolve();frozen=json.loads((out/"freeze.json").read_text())
    matrix=json.loads((out/"matrix.json").read_text())
    results=out/"results";results.mkdir(exist_ok=False)
    status={"status":"RUNNING","started":now(),"steps":[],"failures":[],"skipped":[]}
    records={};invalid=None
    status["runtime"]={"host":platform.platform(),"gpu":capture(["nvidia-smi"]),
       "environment":{k:os.environ.get(k) for k in ("WM_PROJECT_DIR","WM_OPTIONS","FOAM_LIBBIN","LD_LIBRARY_PATH","OMP_NUM_THREADS","CUDA_VISIBLE_DEVICES")}}
    try:
        for rel,expected in json.loads((out/"matrix-sha256.json").read_text()).items():
            assert digest(out/rel)==expected,"Changed frozen matrix "+rel
        for rel,expected in json.loads((out/"frozen-sha256.json").read_text()).items():
            assert digest(out/rel)==expected,"Changed frozen input "+rel
        for index,node in enumerate(matrix["nodes"]):
            rec=dict(node,started=now());status["steps"].append(rec);records[node["id"]]=rec
            directory=results/f'{index:03d}-{node["id"]}';directory.mkdir()
            missing=[key for key in node["dependencies"] if records.get(key,{}).get("result")!="RUN_PASS"]
            affected=invalid and (invalid["scope"]=="all" or node["gpu"])
            if missing or affected:
                rec.update(result="SKIP_DEPENDENCY",reason=invalid if affected else {"dependencies":missing})
                status["skipped"].append(node["id"])
            else:
                try:
                    if node["kind"]=="comparison":rec["comparison"]=compare(node,records);rec["result"]="RUN_PASS"
                    else:
                        root=out/"source"/node["variant"]
                        env=dict(os.environ,UGKP_SOURCE_ROOT=str(root),UGKP_ACCEPTANCE_BIN=str(root/"bin"),
                                 PYTHONDONTWRITEBYTECODE="1")
                        stop=threading.Event();monitor=threading.Thread(target=gpu_samples,args=(stop,directory/"gpu.csv"))
                        monitor.start();start=time.perf_counter()
                        try:
                            with (directory/"stdout.log").open("w") as stream:
                                proc=subprocess.run(node["argv"],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                            rec["exit"]=proc.returncode
                        finally:
                            stop.set();monitor.join();rec["wall_seconds"]=time.perf_counter()-start
                        rec["result"]="RUN_PASS" if rec["exit"]==0 else "FAIL"
                        if rec["result"]=="RUN_PASS" and node["kind"] in ("correctness","performance"):
                            rec["payload"]=payload(directory/"stdout.log")
                            if node["kind"]=="performance":
                                samples=rec["payload"]["samples_seconds"]
                                assert len(samples)==5 and all(math.isfinite(x) and x>0 for x in samples)
                                rec["timing"]={"samples_seconds":samples,"median_seconds":statistics.median(samples),
                                               "min_seconds":min(samples),"max_seconds":max(samples)}
                        if node["kind"]=="native":
                            report=Path(node["argv"][-1])/"results.json"
                            if report.is_file():rec["native"]=json.loads(report.read_text())
                except Exception as error:
                    rec.update(result="FAIL",error=repr(error),traceback=traceback.format_exc())
                if rec["result"]=="FAIL":
                    text=(directory/"stdout.log").read_text(errors="replace") if (directory/"stdout.log").exists() else rec.get("error","")
                    invalid=invalid or failed_environment_scope(text,rec.get("exit"),node["gpu"])
                    status["failures"].append(node["id"])
            rec["finished"]=now();save(directory/"execution.json",rec)
            save(out/"batch-status.json",status)
        status["status"]="PASS_EXECUTION_REVIEW_REQUIRED" if not status["failures"] and not status["skipped"] else "FAIL_COLLECTED_REVIEW_REQUIRED"
    except BaseException as error:
        status.update(status="FAIL_STOPPED",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status["finished"]=now()
        status["limits"]=["All tests are new executions against CP2 baseline and CP3 candidate.",
          "CPU fixture timing is host arithmetic, not GPU throughput. Native wall time is not a benchmark.",
          "GPU telemetry is device-wide, possibly unavailable on WSL; no temperature/load rejection gate.",
          "Native short windows do not establish production steady-state convergence.",
          "No radiation target-residual claim is made; historical CP2 CHT reported 4.909006e-7 > requested 1e-8."]
        save(out/"batch-status.json",status)
        save(out/"evidence-sha256.json",{str(p.relative_to(out)):digest(p) for p in results.rglob("*") if p.is_file()})
    summarizer=out/"source/candidate/validation/checkpoint03/summarize.py"
    with (out/"summary.log").open("w") as stream:
        report=subprocess.run(["python3",str(summarizer),str(out)],stdout=stream,stderr=subprocess.STDOUT)
    status["summary_exit"]=report.returncode;save(out/"batch-status.json",status)
    raise SystemExit(0 if status["status"]=="PASS_EXECUTION_REVIEW_REQUIRED" else 1)
def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    f=sub.add_parser("freeze");f.add_argument("--baseline",type=Path,required=True)
    f.add_argument("--candidate",type=Path,required=True);f.add_argument("--compiled",type=Path,required=True)
    f.add_argument("--output",type=Path,required=True)
    r=sub.add_parser("run");r.add_argument("batch",type=Path)
    a=p.parse_args();(freeze if a.action=="freeze" else run)(a)
if __name__=="__main__":main()
