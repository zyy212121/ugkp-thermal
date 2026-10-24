#!/usr/bin/env python3
"""Native CHMT verification and repeatable execution timing (fresh outputs only)."""
import argparse,csv,hashlib,importlib.util,json,math,os,re,subprocess,time
from pathlib import Path

def run(command,log,env):
    start=time.perf_counter()
    with log.open('w') as stream: subprocess.run([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
    return time.perf_counter()-start

def rows(path):
    with path.open() as stream:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]

def inventory(case):
    out=case/'chmtOutput';steps=sorted(int(p.stem.split('-')[1]) for p in out.glob('gas-*.csv'))
    states=[]
    for step in (steps[0],steps[-1]):
        g=rows(out/f'gas-{step}.csv');s=rows(out/f'material-{step}.csv')
        assert all(math.isfinite(v) for r in g+s for v in r.values())
        assert all(v>=-1e-12 for r in s for k,v in r.items() if k.startswith('mass_'))
        states.append(dict(mass=math.fsum(r['mass'] for r in g)+math.fsum(v for r in s for k,v in r.items() if k.startswith('mass_')),
            energy=math.fsum(r['total_energy'] for r in g+s),solid0=math.fsum(r['mass_C0'] for r in s),
            volume=math.fsum(r['volume'] for r in s)))
    a,b=states;mr=abs(b['mass']-a['mass'])/a['mass'];er=abs(b['energy']-a['energy'])/abs(a['energy'])
    assert mr<1e-10,(case,mr);assert er<1e-10,(case,er)
    return dict(initial=a,final=b,mass_relative=mr,energy_relative=er)

def uniform(path,value):
    text=path.read_text();text,n=re.subn(r'internalField\s+uniform\s+[^;]+;',f'internalField uniform {value};',text)
    assert n==1,path;path.write_text(text)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--scenarios',nargs='+',default=['thermal','single','diffusion','dry','receding']);p.add_argument('--grid',type=int,default=2)
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--memcheck',action='store_true')
    p.add_argument('--refine',action='store_true');p.add_argument('--end-time',type=float);p.add_argument('--restart-from',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);env=dict(os.environ,UGKP_SOURCE_ROOT=str(a.source_root),UGKP_CASES_ROOT=str(a.cases_root))
    fixture=a.cases_root/'test/tools/CHMT/coupled_runtime';results=[]
    for name in a.scenarios:
        for repeat in range(a.repeat):
            case=a.output/f'{name}-{repeat}'
            if name=='receding':
                command=['python3',a.cases_root/'test/tools/chmt_wall_suite/generate.py','chmt_receding_slab','--output',case]
            else:
                command=['python3',fixture/'make_case.py',case]+(['--'+name] if name in ('single','diffusion') else [])
                if a.refine:command+=['--refine']
                if a.end_time is not None:command+=['--end-time',str(a.end_time)]
                if a.restart_from:command+=['--restart',a.restart_from]
            run(command,a.output/f'{name}-{repeat}-generate.log',env)
            props=case/'constant/chmtProperties'
            if a.grid!=2:
                assert name in ('thermal','single','dry')
                for f in ('blockMeshDict','blockMeshSolidDict'):
                    path=case/'system'/f;path.write_text(path.read_text().replace('(2 2 2)',f'({a.grid} {a.grid} {a.grid})'))
                spec=json.loads((case/'verification.json').read_text());spec.update(gas_cells=a.grid**3,material_cells=a.grid**3,interface_faces=a.grid**2)
                (case/'verification.json').write_text(json.dumps(spec))
            if name=='dry':
                # Calibrated mass-concentration first-order bulk reaction; all
                # generated gas species use the production shared calorics.
                text=props.read_text();text+='''
enableReactions true;
permeability 1e-14; poreViscosity 1e-5; poreEffectiveDiffusivity (0.0001 0.0002);
reactions { pyrolysis { A 2; temperaturePower 0; activationEnergy 0; condensedNu (-1 0.3); gasNu (0.2 0.5); order (1 0); } }
material { enablePoreOutflow true; }
''';props.write_text(text)
                uniform(case/'0/solid/rho_C0',500);uniform(case/'0/solid/porosity',.5)
                uniform(case/'0/solid/rhoPore_S0',.3);uniform(case/'0/solid/rhoPore_S1',.2)
                R=8.31446261815324
                energy=(500*1000+.3*(1040-R/.028)+.2*(1040-R/.032))*300
                uniform(case/'0/solid/solidEnergyDensity',format(energy,'.17g'))
                extent=case/'0/solid/reactionExtentDensity_0';text=(case/'0/solid/rho_C1').read_text().replace('rho_C1','reactionExtentDensity_0').replace('[1 -3 0 0 0 0 0]','[0 -3 0 0 1 0 0]');extent.write_text(text)
            run(['blockMesh','-case',case],case/'mesh-gas.log',env)
            run(['blockMesh','-case',case,'-region','solid','-dict',case/'system/blockMeshSolidDict'],case/'mesh-solid.log',env)
            command=[a.binary,'-case',case]
            if a.memcheck:command=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']+command
            elapsed=run(command,case/'run.log',env)
            result=dict(scenario=name,repeat=repeat,elapsed_seconds=elapsed,case=str(case),**inventory(case))
            if name in ('thermal','single','diffusion'):
                checker=a.source_root/'validation/openfoam10_dry/native/check_coupled.py'
                run(['python3',checker,case],case/'independent-check.json',env)
            if name=='dry':
                exact=500*math.exp(-2*(a.end_time or .001))
                error=abs(result['final']['solid0']-exact)
                assert error<.002,error;result['pyrolysis_mass_error']=error
            if name=='receding':
                spec=json.loads((case/'suite_case.json').read_text());rate=spec['parameters']['surface_mass_flux_kg_m2_s']*spec['parameters']['surface_area_m2']
                removed=result['initial']['solid0']-result['final']['solid0'];expected=rate*spec['end_time']
                assert abs(removed-expected)<1e-10,(removed,expected)
                volume_loss=result['initial']['volume']-result['final']['volume']
                assert abs(volume_loss-expected/1000)<1e-12
                result.update(surface_mass_error=removed-expected,recession_volume_error=volume_loss-expected/1000)
            results.append(result);print(json.dumps(result),flush=True)
            (a.output/'results.json').write_text(json.dumps(dict(binary=str(a.binary),sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),grid=a.grid,
                cuda_stack_bytes=env.get('CHMT_CUDA_STACK_BYTES'),memcheck=a.memcheck,results=results),indent=2)+'\n')
if __name__=='__main__':main()
