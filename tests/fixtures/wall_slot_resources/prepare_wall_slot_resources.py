"""Build and run actual CUDA wall-slot resource regression with optional memcheck."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

def validate_run_output(output):
    paths=('full','mass-exchange','mass-supplied-null','mass-supplied-unready')
    failures=re.findall(r'^PASS wall-slot path=(\S+) native=(\d+) count=(-?\d+) slot=(-?\d+) invalidStorage=1 unchangedPhysical=1$',output,re.M)
    expected=[(path,str(native),str(count),str(slot)) for path in paths for native in (0,1) for count,slot in ((2,2),(2,9),(0,0))]
    if sorted(failures)!=sorted(expected):raise ValueError('incomplete invalid-slot matrix')
    positive=re.findall(r'^PASS wall-last path=(\S+) native=(\d+) unusedOptional=1 mass=\S+ energy=\S+$',output,re.M)
    if sorted(positive)!=sorted((path,str(native)) for path in paths for native in (0,1)):raise ValueError('incomplete last-slot control matrix')
    if output.splitlines().count('PASS real CUDA wall-slot resources invalid=24 valid=8 retries=24')!=1:raise ValueError('missing completed resource matrix')
    return dict(invalid=24,valid=8,retries=24)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--generate-only',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--memcheck',action='store_true');parser.add_argument('--nvcc',default=shutil.which('nvcc'))
    parser.add_argument('--arch',default=os.environ.get('UGKWP_CUDA_ARCH','sm_89'))
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    base=output/'native_fixture'
    subprocess.run([sys.executable,str(root/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py'),str(root),str(base),'--generate-only'],check=True,capture_output=True,text=True)
    original=(base/'chemistry_trial.cu').read_text()
    prefix=original.split('bool sameBits(double x,double y)',1)[0].replace('namespace chemistryTrialTest {','namespace wallSlotResources {')
    if 'DeviceState* fixture()' not in prefix or prefix==original:raise ValueError('native fixture boundary changed')
    source=output/'wall_slot_resources.cu';source.write_text(Path(__file__).with_name('wall_slot_resources.cu.in').read_text().replace('FIXTURE_PREFIX',prefix))
    binary=output/'wall_slot_resources';native=root/'applications/gasUGKP/private_backend'
    command=[args.nvcc or 'nvcc','-std=c++17','-O3','-arch='+args.arch,'--fmad=true','-I'+str(native),'-I'+str(root/'common'),'-I'+str(root/'common/gasNumerics'),'-I'+str(root/'applications/gasUGKP/gpu'),str(source),'-o',str(binary)]
    (output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    dependencies=[root/'common/operators/computeRiemannGasFaceFluxDevice.cuh',root/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py',root/'tests/fixtures/chemistry_trial/chemistry_trial.cu.in',Path(__file__),Path(__file__).with_name('wall_slot_resources.cu.in')]
    (output/'source-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},indent=2)+'\n')
    if args.generate_only:print(source);return
    if not args.nvcc:parser.error('nvcc unavailable; actual CUDA is required')
    with (output/'build.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
    print(binary,flush=True)
    if args.run or args.memcheck:
        commands=[('run',[str(binary)])]
        if args.memcheck:commands.append(('memcheck',[shutil.which('compute-sanitizer') or 'compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(binary)]))
        results=[]
        for name,command in commands:
            with (output/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            text=(output/(name+'.log')).read_text();row=dict(name=name,command=command,exit_code=result.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            if result.returncode==0:
                row.update(validate_run_output(text))
                if name=='memcheck' and 'ERROR SUMMARY: 0 errors' not in text:raise ValueError('memcheck did not complete cleanly')
            results.append(row);(output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            if result.returncode:raise SystemExit(result.returncode)
        print(json.dumps(results,indent=2))

if __name__=='__main__':main()
