#!/usr/bin/env python3
"""Reproduce this follow-up in TEST only; GPU runs are opt-in."""
import argparse,hashlib,json,os,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path,required=True);p.add_argument('--suite',choices=['affected','wall','interface'],default='affected');p.add_argument('--scan',action='store_true');p.add_argument('--gpu',action='store_true');p.add_argument('--sanitizers',action='store_true');p.add_argument('--build',action='store_true');a=p.parse_args();root=a.root.resolve();out=a.output.resolve();formal=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal').resolve()
if root==formal or out==formal or formal in out.parents:raise SystemExit('Production is read-only: choose a separate TEST source/output.')
if not(root/'common/gasWall/ReactingWallLayer.H').is_file():raise SystemExit('Not a solver source checkout')
out.mkdir(parents=True,exist_ok=False);runtime=root/'.review-tmp'/('state-reproduce-'+hashlib.sha256(str(out).encode()).hexdigest()[:12]);env={**os.environ,'REVIEW_SOURCE':str(root),'REVIEW_OUTPUT':str(out),'REVIEW_RUNTIME':str(runtime)};records=[]
wall=['test_wall_trial_composition.py','test_wall_newton_globalization.py','test_wall_feasible_derivatives.py','test_wall_jacobian_scale.py','test_wall_team_equivalence.py','test_cuda_wall_team_contract.py'];tests=(['validation/tests/gas_wall/'+x for x in wall]if a.suite!='interface'else[])+(['validation/tests/test_interface_composition.py','validation/tests/test_interface_catalogue_provenance.py']if a.suite!='wall'else[])
setup="""source /opt/openfoam10/etc/bashrc
set -eo pipefail
ulimit -s 65536
export FOAM_USER_APPBIN="$REVIEW_RUNTIME/install/bin" FOAM_USER_LIBBIN="$REVIEW_RUNTIME/install/lib"
export CUDA_HOME=/usr/local/cuda-13.1 NVCC=/usr/local/cuda-13.1/bin/nvcc
export PATH="$FOAM_USER_APPBIN:$FOAM_APPBIN:$CUDA_HOME/bin:/usr/bin:/bin:$WM_PROJECT_DIR/wmake"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN/dummy:$CUDA_HOME/lib64"
export CUDA_CACHE_PATH="$REVIEW_RUNTIME/cache/cuda" XDG_CACHE_HOME="$REVIEW_RUNTIME/cache" TMPDIR="$REVIEW_RUNTIME/tmp"
export UGKWP_GAS_SPECIES=2 CHMT_CUDA_ARCH=sm_89 UGKWP_CUDA_ARCH=sm_89
export PYTHONPATH="$REVIEW_SOURCE/validation/tests/gas_transport:$REVIEW_SOURCE/validation/tests/gas_wall:$REVIEW_SOURCE"
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$REVIEW_SOURCE"
"""
def run(name,cmd):
 script=out/(name+'.sh');script.write_text(setup+'\n'+shlex.join(list(map(str,cmd)))+'\n')
 with(out/(name+'.log')).open('w')as log:q=subprocess.run(['bash',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'command':list(map(str,cmd)),'exit':q.returncode});(out/'commands.json').write_text(json.dumps(records,indent=2));print(name,q.returncode,flush=True)
 if q.returncode:raise SystemExit(q.returncode)
if a.gpu:
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:name=Path(proc.read_bytes().split(b'\0')[0].decode()).name
  except(OSError,UnicodeDecodeError):continue
  if name in ['gasUGKP','gasUGKPCudaBackend','FSH','FSHCudaBackend','CHT','CHMT']:raise SystemExit('Production solver active; postpone GPU tests.')
 q=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi'],capture_output=True,text=True);(out/'gpu-preflight.log').write_text(q.stdout+q.stderr)
else:env['CUDA_VISIBLE_DEVICES']=''
if a.build:
 for name,path in [('gas','tool/gasUGKP/build_private_backend.sh'),('fsh','tool/FSH/build_private_backend.sh'),('cht','tool/CHT/build_cuda_solver.sh'),('chmt','applications/CHMT/Allwmake')]:run('build-'+name,['bash',path])
run('tests',['python3','-m','pytest','-q',*tests,'--basetemp',runtime/'pytest','--junitxml',out/'results.xml'])
if a.scan:
 run('scan-build',['g++','-std=c++17','-O2','-ffp-contract=off','-DDIAGNOSTICS=1','-I'+str(root),Path(__file__).parent/'fixtures/wall_scan.cpp','-o',out/'wall-scan'])
 run('scan',[out/'wall-scan','1000'])
if a.gpu:
 gpu=[]
 if a.suite!='interface':gpu += [('wall-composition','validation/tests/gas_wall/fixtures/wall_composition_gpu.cu',None),('captured-sst','validation/tests/gas_wall/fixtures/sst_globalization_gpu.cu',None),('trial-storage','validation/tests/gas_wall/fixtures/wall_trial_storage_gpu.cu',None)]
 if a.suite!='wall':gpu += [('interface2','validation/tests/fixtures/interface_composition/device.cu',2),('interface10','validation/tests/fixtures/interface_composition/device.cu',10)]
 for name,file,ns in gpu:
  cmd=['/usr/local/cuda-13.1/bin/nvcc','-std=c++17','-O3','-lineinfo','--fmad=false','-arch=sm_89','-I'+str(root),'-I'+str(root/'applications/CHMT'),'-I'+str(root/'common')]+(['-DUGKWP_GAS_SPECIES='+str(ns)]if ns else[])+[root/file,'-o',out/name]
  run(name+'-build',cmd);run(name,[out/name])
  if a.sanitizers:
   for tool in ['memcheck','racecheck','synccheck']:run(name+'-'+tool,['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','77',out/name])
record={'runtime_ext4':str(runtime),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in ['common/gasWall/ReactingWallLayer.H','common/DiagnosticsCatalogue.H','applications/CHMT/ablation/InterfaceComposition.H']},'records':records};(out/'reproduction.json').write_text(json.dumps(record,indent=2))
