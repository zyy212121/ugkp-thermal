#!/usr/bin/env python3
import argparse,hashlib,json,os,re,shutil,statistics,subprocess,time
from pathlib import Path
def run(cmd,log,env):
    t=time.perf_counter()
    with log.open('w') as f:subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
    return time.perf_counter()-t
def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--repeat',type=int,default=5);a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=False);env=dict(os.environ,UGKP_SOURCE_ROOT=str(a.source),GAS_UGKP_CUDA_STACK_BYTES='65536')
    generate=a.cases/'test/tools/chmt_wall_suite/generate.py'
    cases={'gas_species_wave':dict(cells=128,end_time=.002),'small_sod':dict(cells=200),'flatplate_fixed':dict(nx=32,ny=16,end_time=.0002,wall='lowRe')}
    report={};allrows=[]
    for name,options in cases.items():
        template=out/(name+'-input');opts=out/(name+'-options.json');opts.write_text(json.dumps(options))
        run(['python3',generate,name,'--output',template,'--options',opts],out/(name+'-generate.log'),env)
        run(['blockMesh','-case',template],out/(name+'-mesh.log'),env)
        if name=='small_sod':run(['setFields','-case',template],out/(name+'-init.log'),env)
        times={'baseline':[],'optimized':[]};hashes={};steps={}
        # Alternate order to reduce warmup/temperature/order bias; discard pair 0.
        for i in range(a.repeat+1):
            for kind in (['baseline','optimized'] if i%2==0 else ['optimized','baseline']):
                case=out/f'{name}-{kind}-{i}';shutil.copytree(template,case)
                binary=a.before if kind=='baseline' else a.after
                elapsed=run([binary,'-case',case],case/'run.log',env)
                text=(case/'run.log').read_text();steps[kind]=len(re.findall(r'^Time = ',text,re.M)) or None
                dirs=[q for q in case.iterdir() if q.is_dir() and re.fullmatch(r'[0-9.eE+-]+',q.name)]
                final=max(dirs,key=lambda q:float(q.name));expected=json.loads((template/'suite_case.json').read_text())['end_time']
                assert abs(float(final.name)-expected)<1e-12,(final,expected)
                fields={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in final.iterdir() if f.is_file() and (f.name in ('rho','p','T','U','rhoE','k','omega') or f.name.startswith('Y_'))}
                assert fields
                if kind in hashes:assert fields==hashes[kind],(name,kind,'nondeterministic fields')
                hashes[kind]=fields
                if i:times[kind].append(elapsed)
                allrows.append(dict(case=name,kind=kind,repetition=i,wall_seconds=elapsed,steps=steps[kind],warmup=i==0))
                print(json.dumps(allrows[-1]),flush=True)
        assert hashes['baseline']==hashes['optimized'],(name,'field difference')
        before=statistics.median(times['baseline']);after=statistics.median(times['optimized'])
        report[name]=dict(seconds=times,median_before=before,median_after=after,speedup=before/after,field_sha256=hashes['baseline'],steps=steps)
        (out/'results.json').write_text(json.dumps(dict(cases=report,runs=allrows),indent=2)+'\n')
if __name__=='__main__':main()
