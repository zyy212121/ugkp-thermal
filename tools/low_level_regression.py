#!/usr/bin/env python3
"""Build the low-level correctness regressions offline; GPU execution is opt-in.

Default and --build-only never initialize CUDA or query GPU state. --integration
also compiles the full-backend survivor/moment/pressure/commit fixtures.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--build-only',action='store_true')
parser.add_argument('--integration',action='store_true')
parser.add_argument('--run',action='store_true',help='Explicitly execute GPU correctness tests after building')
args=parser.parse_args()
if args.build_only and args.run:parser.error('--build-only and --run conflict')
out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
resource.setrlimit(resource.RLIMIT_STACK,(512*1024*1024,resource.RLIM_INFINITY))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
nvcc=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'
arch=os.environ.get('UGKWP_CUDA_ARCH','sm_89')
rows=[];executables=[]
def build(label,cmd,exe):
    log=out/(label+'.build.log')
    with log.open('w') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(test=label,command=cmd,exit=q.returncode,executable=str(exe),gpu_executed=False))
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(label,'BUILD',q.returncode,flush=True)
    if q.returncode:print(log.read_text()[-5000:]);sys.exit(1)
    executables.append((label,exe))
fixture=ROOT/'tests/fixtures/low_level'
for label,real,pruned,fmad in [('gas64','double',1,True),('FSH64','double',0,True),('CHT64','double',0,False),('CHT32','float',0,False)]:
    d=out/label;d.mkdir(exist_ok=True)
    text=(fixture/'reduction.cu.in').read_text()
    for key,value in {'REAL':real,'PRUNED':str(pruned),'LABEL':'gas' if pruned else 'thermal'}.items():text=text.replace('@'+key+'@',value)
    source=d/'reduction.cu';source.write_text(text);exe=d/'reduction'
    build(label+'-bitwise',[str(nvcc),'-std=c++17','-O3','-arch='+arch,'--fmad='+str(fmad).lower(),'-I'+str(ROOT/'common'),'-I'+str(fixture),str(source),'-o',str(exe)],exe)
for app,bits in [('FSH',64),('CHT',64),('CHT',32)]:
    label=app+str(bits);d=out/label;real='float' if bits==32 else 'double'
    thermal=(ROOT/'common/GpuCellLocalThermalFields.cuh').read_text()
    primary=(ROOT/'common/GpuCellLocalPrimary.cuh').read_text()
    hook=thermal+'\n#define GPU_PARTICLE_EXTRA_FIELDS CellLocalThermalExtraFields<'+('false, false' if app=='FSH' else 'true, true')+'>\n'+primary
    source=d/'payload.cu';source.write_text((ROOT/f'tests/fixtures/thermal_workers/gather_{app}.cu.in').read_text().replace('@PARTICLE_COPY_HOOK@',hook).replace('@REAL_TYPE@',real));exe=d/'payload'
    build(label+'-payload',[str(nvcc),'-std=c++17','-O3','-arch='+arch,'--fmad='+str(app=='FSH').lower(),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],exe)
    for generator in ['tests/fixtures/shared_operators/payload_commit.py']+(['tests/fixtures/thermal_workers/s1_pipeline.py'] if args.integration else []):
        target=d/Path(generator).stem;env=dict(os.environ,UGKP_CUDA_BUILD_ONLY='1')
        cmd=[sys.executable,str(ROOT/generator),str(ROOT),str(target),app,str(bits)]
        log=out/(label+'-'+Path(generator).stem+'.build.log')
        with log.open('w') as f:q=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        matches=list(target.rglob('payload_commit' if 'payload_commit' in generator else 'exact13_*'))
        matches=[p for p in matches if p.is_file() and p.suffix=='']
        rows.append(dict(test=label+'-'+Path(generator).stem,command=cmd,exit=q.returncode,executable=str(matches[0]) if matches else None,gpu_executed=False))
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1]['test'],'BUILD',q.returncode,flush=True)
        if q.returncode or len(matches)!=1:print(log.read_text()[-5000:]);sys.exit(1)
        executables.append((rows[-1]['test'],matches[0]))
# Record all production inputs and fixtures so the deferred run is tied to this
# source state. Build logs contain exact command lines/flags in results.json.
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for scope in ['applications','common','gpu','tests/fixtures']
    for p in (ROOT/scope).rglob('*') if p.is_file() and p.suffix in {'.cu','.cuh','.H','.in','.inl','.py'} and '__pycache__' not in p.parts}
(out/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
if args.run:
    for label,exe in executables:
        q=subprocess.run([str(exe)],capture_output=True,text=True)
        (out/(label+'.run.log')).write_text(q.stdout+q.stderr)
        row=next(r for r in rows if r['test']==label);row.update(gpu_executed=True,run_exit=q.returncode)
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(label,'RUN',q.returncode,flush=True)
        if q.returncode:print(q.stdout+q.stderr);sys.exit(1)
else:print('BUILD ONLY complete. GPU correctness execution remains pending. No performance timing.',flush=True)
