#!/usr/bin/env python3
"""Compile-only. Never execute a solver, generated binary, mathematical test, or benchmark."""
from pathlib import Path
import argparse, ast, hashlib, json, os, shutil, subprocess, time
FAMILIES=("pressure","face","coupling","bath","probe")
UNUSED_CONTROL="validation/checkpoint03/prepare.py"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=("baseline","candidate"),required=True)
    p.add_argument("--families",nargs="+",choices=FAMILIES,default=FAMILIES)
    p.add_argument("--bits",type=int,nargs="+",choices=(32,64),default=(64,32))
    p.add_argument("--arch",default="sm_89")
    p.add_argument("--revision",default="01")
    p.add_argument("--rebuild",action="store_true",help="Explicit compile-only replacement; retain prior artifacts and records")
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((out/"compiled.json").read_text()) if (out/"compiled.json").is_file() else dict(
        status="PREPARING_NOT_RUN",programs={},variants={},commands=[])
    root=(a.baseline if a.variant=="baseline" else a.candidate).resolve()
    controls=Path(__file__).resolve().parent;cuda=Path(os.environ.get("CUDA_HOME","/usr/local/cuda"))
    def command(argv,name):
        argv=list(map(str,argv));log=out/(name+".build.log")
        assert not log.exists(),"Do not overwrite compile evidence; use a new output directory for a new recipe"
        record=dict(argv=argv,log=log.name,started=time.time());meta["commands"].append(record);save(out/"compiled.json",meta)
        with log.open("w") as stream:r=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
        record.update(exit=r.returncode,finished=time.time());save(out/"compiled.json",meta)
        if r.returncode:raise RuntimeError("Compilation failed; see "+str(log))
    for bits in a.bits:
        materialized=root/".build/precision"/f"fp{bits}"/"source"
        manifest_path=materialized.parent/"source-manifest.json"
        manifest=json.loads(manifest_path.read_text())
        source_hashes={name:digest(materialized/name) for name in manifest["files"] if name!=UNUSED_CONTROL}
        info={"tree":str(materialized),"manifest":manifest,"source_sha256":source_hashes,
              "source_exclusions":{UNUSED_CONTROL:"Not imported or included. Actual preparation argv and copied fixture source hashes are recorded separately."}}
        old=meta["variants"].setdefault(a.variant,{}).get(str(bits))
        assert old is None or old["source_sha256"]==source_hashes,"Source changed across compile preparation"
        meta["variants"][a.variant][str(bits)]=info
        dest=out/a.variant/f"fp{bits}";dest.mkdir(parents=True,exist_ok=True)
        host=["g++","-std=c++17","-O2","-ffp-contract=off","-I"+str(materialized/"common"),
            "-I"+str(materialized/"common/gasNumerics"),"-I"+str(materialized/"common/gpu"),
            "-I"+str(materialized/"validation/tests/gas_transport"),"-I"+str(cuda/"include")]
        cu=[str(cuda/"bin/nvcc"),"--Ofast-compile=min","-std=c++17","-O2","--fmad=false",
            "--expt-relaxed-constexpr","-arch="+a.arch,"-I"+str(materialized/"common"),
            "-I"+str(materialized/"common/gasNumerics"),"-I"+str(materialized/"common/gpu")]
        for family in a.families:
            key=f"{a.variant}-fp{bits}-{family}"
            if key in meta["programs"]:
                assert a.rebuild,"Program already compiled; explicit --rebuild required"
            buildkey=key+"-v"+a.revision
            work=dest/(family+"-v"+a.revision);work.mkdir()
            prior=meta["programs"].get(key)
            if family=="face":
                for name in ("generate_face.py","face_fixture.cpp"):shutil.copy2(controls/name,work/name)
                command(["python3",work/"generate_face.py","--root",materialized,"--output",work/"generated.cpp"],buildkey+"-generate")
                source=work/"generated.cpp";compiler=host
            elif family=="probe":
                source=work/"probe.cpp"
                tree=ast.parse((materialized/"validation/pr41/native_injection.py").read_text())
                body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="PROBE" for t in n.targets))
                source.write_text(body);compiler=host+["-I"+str(materialized/"applications/CHMT")]
            else:
                name=family+"_fixture."+("cu" if family=="pressure" else "cpp")
                source=work/name;shutil.copy2(controls/name,source);compiler=cu if family=="pressure" else host
            binary=work/family
            if family=="coupling":
                if bits==64:compiler=compiler+["-DCP3_FP64"]
                if a.variant=="candidate":compiler=compiler+["-DCP3_CELL_PREPARES_HEAT"]
            uninstrumented=list(compiler)
            if family=="pressure":
                compiler=compiler+["-DCP3_COUNT_PRESSURE"]
                if a.variant=="candidate":compiler=compiler+["-DCP3_EXPECT_SHARED"]
            command(compiler+[source,"-o",binary],buildkey)
            performance_binary=binary
            if family=="pressure":
                performance_binary=work/"pressure-perf"
                command(uninstrumented+[source,"-o",performance_binary],buildkey+"-perf")
            if prior:meta.setdefault("superseded_programs",[]).append(dict(id=key,program=prior))
            meta["programs"][key]={"executable":str(binary.relative_to(out)),"sha256":digest(binary),
                "performance_executable":str(performance_binary.relative_to(out)),"performance_sha256":digest(performance_binary),
                "gpu":family=="pressure","source":str(source.relative_to(out)),"source_sha256":digest(source),
                "fixture_sources":{p.name:digest(p) for p in work.iterdir() if p.suffix in (".cpp",".cu",".py",".hpp")},
                "correctness_args":["correctness"],"performance_args":["perf"]}
            save(out/"compiled.json",meta)
        assert source_hashes=={name:digest(materialized/name) for name in source_hashes},"Source changed while compiling"
    expected={f"{v}-fp{b}-{f}" for v in ("baseline","candidate") for b in (32,64) for f in FAMILIES}
    meta["status"]="COMPILED_NOT_RUN" if expected<=meta["programs"].keys() else "PREPARING_NOT_RUN"
    save(out/"compiled.json",meta);print(meta["status"]+"; no numerical execution.")
if __name__=="__main__":main()
