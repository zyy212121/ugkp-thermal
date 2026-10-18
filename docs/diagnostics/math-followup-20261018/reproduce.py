#!/usr/bin/env python3
"""Reproduce affected tests in an isolated output directory; never install to production."""
import argparse,datetime,json,os,subprocess,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases-root',type=Path);p.add_argument('--build',action='store_true');p.add_argument('--gpu',action='store_true');a=p.parse_args();r=a.root.resolve();out=a.output.resolve()
formal=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal').resolve()
if r==formal or out==formal or formal in out.parents:raise SystemExit('Use a separate test checkout/output; the production checkout is read-only.')
if not (r/'common/Diagnostics.H').is_file():raise SystemExit('Invalid source checkout')
out.mkdir(parents=True,exist_ok=False)
env='source /opt/openfoam10/etc/bashrc\nset -eo pipefail\nexport FOAM_USER_APPBIN="$REVIEW_OUTPUT/install/bin"\nexport FOAM_USER_LIBBIN="$REVIEW_OUTPUT/install/lib"\nexport CUDA_HOME=/usr/local/cuda-13.1\nexport NVCC="$CUDA_HOME/bin/nvcc"\nexport PATH="$FOAM_USER_APPBIN:$FOAM_APPBIN:$CUDA_HOME/bin:/usr/bin:/bin:$WM_PROJECT_DIR/wmake"\nexport LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN/dummy:$CUDA_HOME/lib64"\nexport CUDA_CACHE_PATH="$REVIEW_OUTPUT/cache/cuda" XDG_CACHE_HOME="$REVIEW_OUTPUT/cache" TMPDIR="$REVIEW_OUTPUT/tmp"\nexport UGKWP_GAS_SPECIES=2 CHMT_CUDA_ARCH=sm_89 UGKWP_CUDA_ARCH=sm_89\nmkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_CACHE_PATH" "$TMPDIR"\nexport PYTHONPATH="$REVIEW_SOURCE/validation/tests/gas_transport:$REVIEW_SOURCE/validation/tests/gas_wall:$REVIEW_SOURCE"\ncd "$REVIEW_SOURCE"\n'
process_env={**os.environ,'REVIEW_OUTPUT':str(out),'REVIEW_SOURCE':str(r)}
if a.cases_root:process_env['UGKP_CASES_ROOT']=str(a.cases_root.resolve())
if not a.gpu:process_env['CUDA_VISIBLE_DEVICES']=''
else:
    for process in Path('/proc').iterdir():
        if process.name.isdigit():
            try:
                name=Path((process/'cmdline').read_bytes().split(b'\0')[0].decode(errors='replace')).name
                if name in ('gasUGKP','gasUGKPCudaBackend','CHMT','CHT','FSH','FSHCudaBackend'):raise SystemExit('A production solver is active; postpone these GPU tests.')
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
records=[]
def run(name,cmd):
    script=out/(name+'.sh');script.write_text(env+'\n'+shlex.join(cmd)+'\n')
    with (out/(name+'.log')).open('w') as log:q=subprocess.run(['bash',str(script)],env=process_env,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':cmd,'exit':q.returncode});(out/'commands.json').write_text(json.dumps(records,indent=2));print(name,q.returncode,flush=True)
    if q.returncode:raise SystemExit(q.returncode)
if a.build:
    for name,script in [('gas','tool/gasUGKP/build_private_backend.sh'),('fsh','tool/FSH/build_private_backend.sh'),('cht','tool/CHT/build_cuda_solver.sh'),('chmt','applications/CHMT/Allwmake')]:run('build-'+name,['bash',script])
tests=json.loads((Path(__file__).with_name('affected-tests.json')).read_text())
run('affected-tests',['python3','-m','pytest','-q',*tests,'--basetemp',str(out/'pytest-temp'),'--junitxml',str(out/'results.xml')])
# The default hides GPUs explicitly; no CPU emulation of GPU evidence is performed.
# --gpu enables the real device diagnostic test in the suite. Extra wall/composition
# device fixtures and sanitizer commands are preserved in root-evidence.tar.gz.
