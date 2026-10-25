import argparse,importlib.util,json,os,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--repeat',type=int,default=5);p.add_argument('--resume',action='store_true');a=p.parse_args()
out=a.output;out.mkdir(exist_ok=a.resume)
source=a.source;runner=source/'validation/openfoam10_dry/native/run_native.py'
s=importlib.util.spec_from_file_location('compare',a.cases/'test/tools/CHMT/coupled_runtime/compare_restart.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
env=dict(os.environ,CHMT_CUDA_STACK_BYTES='65536');records={};runs=[]
for scenario,grid in [('thermal',8),('dry',8),('receding',2)]:
    samples={'baseline':[],'optimized':[]};reference=None
    for repeat in range(a.repeat+1):
        for kind in (['baseline','optimized'] if repeat%2==0 else ['optimized','baseline']):
            target=out/f'{scenario}-{kind}-{repeat}'
            binary=a.before if kind=='baseline' else a.after
            if not (a.resume and (target/'results.json').exists()):
                if target.exists():target=out/(target.name+'-retry')
                with (out/f'{target.name}.log').open('w') as log:
                    subprocess.run(['python3',str(runner),'--binary',str(binary),'--source-root',str(source),'--cases-root',str(a.cases),'--output',str(target),'--scenarios',scenario,'--grid',str(grid)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
            result=json.loads((target/'results.json').read_text())['results'][0]
            case=target/(scenario+'-0')
            if reference is None:reference=case
            parity=m.compare_cases(reference,case)
            if repeat:samples[kind].append(result['elapsed_seconds'])
            runs.append(dict(scenario=scenario,grid=grid,kind=kind,repeat=repeat,warmup=repeat==0,seconds=result['elapsed_seconds'],state_sha256=parity['state_sha256'],accepted_step=parity['accepted_step']))
            print(json.dumps(runs[-1]),flush=True)
    before=statistics.median(samples['baseline']);after=statistics.median(samples['optimized'])
    records[scenario]=dict(grid=grid,cells_per_region=grid**3,seconds=samples,median_before=before,median_after=after,speedup=before/after,state_payload_bitwise_equal=True)
    (out/'results.json').write_text(json.dumps(dict(cases=records,runs=runs),indent=2)+'\n')
