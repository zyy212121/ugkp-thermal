#!/usr/bin/env python3
"""Compile-only recipe. No fixture, solver or benchmark is executed here."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, subprocess
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--arch",default=os.environ.get("CUDA_ARCH","sm_89"))
    args=p.parse_args(); root=args.root.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cuda=Path(os.environ.get("CUDA_HOME","/usr/local/cuda")); nvcc=str(cuda/"bin/nvcc")
    tasks=[]; commands=[]
    variant_sources={}
    for precision in ("32","64"):
        manifest=root/".build/precision"/("fp"+precision)/"source-manifest.json"
        data=json.loads(manifest.read_text())
        tree=manifest.parent/"source"
        variant_sources[precision]={"manifest":data,"actual_sha256":{
            name:digest(tree/name) for name in data["files"] if (tree/name).is_file()}}
    (out/"variant-sources-before-build.json").write_text(json.dumps(variant_sources,indent=2)+"\n")
    def command(argv,name):
        argv=list(map(str,argv)); commands.append(dict(name=name,argv=argv))
        (out/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
        with (out/(name+".build.log")).open("w") as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    def register(bits,name,exe,args=(),coverage=""):
        tasks.append(dict(id="fp"+str(bits)+"-"+name,precision=bits,
            executable=str(exe.relative_to(out)),args=list(args),coverage=coverage))
    for bits in (64,32):
        tree=root/".build/precision"/("fp"+str(bits))/"source"
        dest=out/("fp"+str(bits)); dest.mkdir()
        cxx=["g++","-std=c++17","-O2","-ffp-contract=off","-I"+str(tree),
             "-I"+str(tree/"common"),"-I"+str(tree/"common/gasNumerics"),
             "-I"+str(cuda/"include")]
        cu=[nvcc,"--Ofast-compile=min","-std=c++17","-O2","--fmad=false",
            "--expt-relaxed-constexpr","-arch="+args.arch,
            "-I"+str(tree),"-I"+str(tree/"common"),"-I"+str(tree/"common/gpu")]
        link=["-L"+str(cuda/"lib64"),"-Wl,-rpath,"+str(cuda/"lib64"),"-lcudart"]
        nvlink=["-L"+str(cuda/"lib64"),"-Xlinker","-rpath","-Xlinker",str(cuda/"lib64"),"-lcudart"]
        def host(name,source,extra=(),run_args=((),),coverage=""):
            exe=dest/name; command(cxx+list(extra)+[source,"-o",exe],"fp"+str(bits)+"-"+name)
            for i,a in enumerate(run_args):
                register(bits,name+("-"+str(i) if len(run_args)>1 else ""),exe,a,coverage)
            return exe
        # Probe is compiled before the batch, never lazily by native postprocessing.
        source=tree/"validation/pr41/native_injection.py"
        probe=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROBE" for t in n.targets))
        probe_file=dest/"source_residual.cpp"; probe_file.write_text(probe)
        probe_exe=dest/"source_residual"
        command(cxx+["-I"+str(tree/"applications/CHMT"),probe_file,"-o",probe_exe],
                "fp"+str(bits)+"-source-residual")
        sparse=dest/"SparseConjugateGradient.o"
        command(cu+["-c",tree/"common/gpu/SparseConjugateGradient.cu","-o",sparse],
                "fp"+str(bits)+"-sparse-object")
        if bits==64:
            host("sparse-legacy",tree/"validation/tests/sparse_conjugate_gradient.cpp",
                 [sparse]+link,coverage="Retained original FP64 sparse operator and true residual.")
            command(["python3",tree/"validation/tests/run_poisson_sample_reduction.py",
                     "--output",dest/"poisson","--arch",args.arch,"--build-only"],"fp64-poisson")
            register(64,"poisson-original",dest/"poisson/poisson-sample-unit",
                     coverage="Original double moments/sampling semantics, not cast to float.")
            exe=dest/"resident-sparse-legacy"
            command(cu+[tree/"validation/tests/resident_sparse_publication.cu",sparse]+nvlink+
                    ["-o",exe],"fp64-resident-sparse")
            register(64,"resident-sparse-legacy",exe,coverage="Original FP64 resident publication/graph lifecycle.")
        else:
            host("sparse-fp32",tree/"validation/tests/fp32_sparse_preconditioning.cpp",
                 [sparse]+link,coverage="FP32 diagonal/z, FP64 true residual and conservation; scale 1e+-150.")
            host("central-host",tree/"validation/tests/fp32_collision_central.cpp",
                 coverage="FP32 weighted central moment host algebra.")
            for thermal in (0,1):
                name="central-"+("thermal" if thermal else "gas"); exe=dest/name
                command(cu+["-DGPU_OPERATOR_THERMAL="+str(thermal),
                        tree/"validation/tests/fp32_collision_central.cu","-o",exe],"fp32-"+name)
                register(32,name,exe,coverage="Production full-cell/pool/sampling/correction; singleton identity.")
            for name in ("double_geometry_intersection","double_wall_source","mixed_chemistry","mixed_eos_rk"):
                host(name,tree/("validation/pr43/"+name+".cpp"),coverage="Retained precision-boundary fixture, rebuilt against FP32 overlay.")
            host("mixed_divergence_ale",tree/"validation/pr43/mixed_divergence_ale.cpp",
                 run_args=tuple((s,) for s in ("static_zero","moving_zero","small_motion",
                     "static_dt_volume","moving_dt_volume","subfloat_dt_volume")))
            for name,thermal in (("mixed_particle_energy",0),("mixed_particle_energy",1),("mixed_coupling_energy",1)):
                host(name+"-"+str(thermal),tree/("validation/pr43/"+name+".cpp"),
                     ["-DFIXTURE_THERMAL="+str(thermal)])
            helper=tree/"validation/pr43/check_fp32_model_regressions.py"
            spec=importlib.util.spec_from_file_location("model_source_only",helper)
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            generated=dest/"material_wall_source.cpp"; generated.write_text(mod.radiation_source())
            host("model-regressions",tree/"validation/pr43/fp32_model_regressions.cpp",
                 coverage="WALE normalization, float epsilon and material-wall model algebra.")
            host("material-wall",generated,coverage="1200K material wall must not become 300K gas boundary.")
        # Same public radiation fixture against each independent operator.
        radobj=dest/"DiscreteOrdinateRadiation.o"; mieobj=dest/"MieOpticalProperties.o"
        for name,obj in (("DiscreteOrdinateRadiation",radobj),("MieOpticalProperties",mieobj)):
            command(cu+["-c",tree/("common/gpu/"+name+".cu"),"-o",obj],
                    "fp"+str(bits)+"-"+name)
        if bits==64:
            host("radiation-original",tree/"validation/pr33/radiation_batches.cpp",
                 [radobj,mieobj]+link,run_args=tuple(("31","8",t) for t in ("isolated","chain","cycle")),
                 coverage="Original strict FP64 radiation, unchanged.")
        host("radiation-pair",root/"validation/checkpoint02/acceptance/radiation_pair.cpp",
             ["-DUGKP_ACCEPTANCE_BITS="+str(bits),radobj,mieobj]+link,
             run_args=tuple(("31","8",t) for t in ("isolated","chain","cycle")),
             coverage="Same declared tolerance and public workload for FP64/FP32 comparison.")
        if bits==32:
            rad=dest/"radiation"
            command(["bash",tree/"validation/checkpoint02/radiation/build_only.sh",rad],
                    "fp32-radiation-detailed")
            register(32,"ordered-radiation",rad/"ordered_radiation",
                     coverage="Ordered dependencies, refinement, independently computed published residual.")
            register(32,"radiation-acceptance",rad/"radiation_acceptance",
                     coverage="Near equilibrium, exact equilibrium, scattering, closure, strict warning, workspace 1/2/8.")
    manifest={"status":"COMPILED_NOT_RUN","root":str(root),"tasks":tasks,
              "probe":{"32":"fp32/source_residual","64":"fp64/source_residual"},
              "variant_sources":variant_sources,
              "files":{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob("*"))
                       if p.is_file()}}
    (out/"fixtures.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Compiled fixtures only. No mathematical, numerical or performance test ran.")
if __name__=="__main__": main()
