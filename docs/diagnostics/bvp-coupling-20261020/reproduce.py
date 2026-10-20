#!/usr/bin/env python3
"""Run affected production-operator checks in a test clone, with isolated outputs."""
from pathlib import Path
import argparse,subprocess,os,json,shlex,hashlib,time
p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
p.add_argument("--build",action="store_true")
p.add_argument("--gpu",action="store_true")
a=p.parse_args();root=Path(__file__).resolve().parents[3];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_CACHE_PATH=str(out/"cuda-cache"),TMPDIR=str(out/"tmp"))
for n in ["bin","lib","tmp","cuda-cache"]:(out/n).mkdir(exist_ok=True)
records=[]
def run(name,cmd,environment=env):
 t=time.time()
 with(out/(name+".log")).open("w")as f:q=subprocess.run(cmd,cwd=root,env=environment,stdout=f,stderr=subprocess.STDOUT)
 record={"name":name,"command":cmd,"exit":q.returncode,"seconds":time.time()-t};records.append(record)
 (out/"commands.json").write_text(json.dumps(records,indent=2))
 if q.returncode:raise SystemExit("Failed "+name+"; inspect its raw log")
tests=[
 "gas_wall/test_wall_conservative_transport.py","gas_wall/test_wall_matching_averages.py",
 "gas_wall/test_wall_matching_state.py","gas_wall/test_wall_power_discretization.py",
 "gas_wall/test_wall_volume_coupling.py","gas_wall/test_wall_energy_balance.py",
 "gas_wall/test_wall_geometry.py","gas_wall/test_wall_publication_work.py",
 "gas_wall/test_wall_sst_audit.py","gas_wall/test_wall_team_equivalence.py",
 "gas_transport/test_boundary_layer_state.py","gas_transport/test_boundary_layer_team.py",
 "gas_transport/test_boundary_layer_volume_projection.py","gas_transport/test_boundary_layer_operators.py"]
run("affected",["python3","-m","pytest","-q",*["validation/tests/"+s for s in tests],"--junitxml="+str(out/"affected.xml")])
if a.build:
 env.update(FOAM_USER_APPBIN=str(out/"bin"),FOAM_USER_LIBBIN=str(out/"lib"),CUDA_HOME="/usr/local/cuda-13.1",UGKWP_CUDA_ARCH="sm_89",UGKWP_GAS_SPECIES="2",GAS_UGKP_FMAD="1")
 script=out/"build.sh"
 exports="".join("export "+key+"="+shlex.quote(env[key])+"\n"for key in ["FOAM_USER_APPBIN","FOAM_USER_LIBBIN","CUDA_HOME","UGKWP_CUDA_ARCH","UGKWP_GAS_SPECIES","GAS_UGKP_FMAD","CUDA_CACHE_PATH","TMPDIR"])
 script.write_text("source /opt/openfoam10/etc/bashrc\nset -eo pipefail\n"+exports+"cd "+shlex.quote(str(root))+"\nbash tool/gasUGKP/build_private_backend.sh\n")
 run("full-build",["bash",str(script)])
if a.gpu:
 exe=out/"wall-volume-pool-gpu"
 run("cuda-compile",["/usr/local/cuda-13.1/bin/nvcc","-std=c++17","-O3","-arch=sm_89","--fmad=true","-I"+str(root),str(root/"validation/tests/gas_wall/fixtures/wall_volume_pool_gpu.cu"),"-o",str(exe)])
 run("cuda-plain",[str(exe)])
 for tool in ["memcheck","racecheck","synccheck"]:
  run("cuda-"+tool,["/usr/local/cuda-13.1/bin/compute-sanitizer","--tool",tool,"--error-exitcode","99",str(exe)])
hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for directory in ["common","applications/gasUGKP"]for f in(root/directory).rglob("*")if f.is_file()and f.suffix in [".H",".C",".cu",".cuh",".inc"]}
(out/"source-hashes.json").write_text(json.dumps(hashes,indent=2))
print("Completed; commands, raw logs and source hashes:",out)
