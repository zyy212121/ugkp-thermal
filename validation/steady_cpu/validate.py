#!/usr/bin/env python3
"""Reproduce CPU mathematical checks without pytest; never certify CUDA parity."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import build


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate(directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    executables={False:build.build(directory/'gas-reference'),True:build.build(directory/'particle-reference',particles=True)}
    result={'cuda_runtime_validated':False,'cases':{}}
    for name in ('reservoir','reconstructed','mixture_sst','mixture_diffusion','particles','mixture_particles'):
        coupled=name in ('particles','mixture_particles')
        cfg=Path(__file__).parent/'cases'/f'{name}.cfg';out=directory/f'{name}.csv'
        subprocess.run([str(executables[coupled]),str(cfg),str(out)],check=True)
        rows=load(out);last=[r for r in rows if r['iteration']==rows[-1]['iteration']]
        record={'input_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'gas_csv_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'rows':len(rows),'cells':len(last),
                'finite':all(math.isfinite(float(v)) for r in rows for v in r.values()),'cell_status_max':max(int(r['cellStatus']) for r in rows),
                'final_max_mass_residual':max(float(r['Rrho']) for r in last)}
        assert record['finite'] and record['cell_status_max']==0
        if name.startswith('mixture'):
            record['max_species_closure_error']=max(abs(float(r['rhoY0'])+float(r['rhoY1'])-float(r['rho'])) for r in rows)
            assert record['max_species_closure_error']<1e-12
        if not coupled:
            record['final_density_error']=max(abs(float(r['rho'])-1) for r in last)
            assert record['final_density_error']<1e-6 and record['final_max_mass_residual']<1e-5
        else:
            parcels=load(Path(str(out)+'.particles.csv'));final=[r for r in parcels if r['iteration']==parcels[-1]['iteration']]
            record['injected']=int(final[0]['injected']);record['outflow']=int(final[0]['outflow'])
            record['particle_mass_budget_error']=float(final[0]['injectedMass'])-float(final[0]['outflowMass'])-sum(float(r['mass']) for r in final)
            assert record['injected']>record['outflow']>0 and abs(record['particle_mass_budget_error'])<1e-13
            windows=load(Path(str(out)+'.windows.csv'))
            assert all(math.isfinite(float(v)) for r in windows for v in r.values())
            if name.startswith('mixture'):
                record['max_window_species_closure_error']=max(abs(float(r['q14Mean'])+float(r['q15Mean'])-float(r['q0Mean'])) for r in windows)
                assert record['max_window_species_closure_error']<1e-12
            repeat=directory/f'{name}-repeat.csv'
            subprocess.run([str(executables[coupled]),str(cfg),str(repeat)],check=True)
            for suffix in ('','.particles.csv','.windows.csv'):
                assert Path(str(out)+suffix).read_bytes()==Path(str(repeat)+suffix).read_bytes()
            record['same_seed_repeat_identical']=True
        record['passed']=True;result['cases'][name]=record;print(name,'PASS',flush=True)
    spec=importlib.util.spec_from_file_location('particle_tests',build.HERE/'steady_cpu_particles.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result['particle_operator_selftest']=module.selftest(build.ROOT,directory/'operator-selftest',sanitizers=True).stdout.strip()
    result['passed']=True
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='/tmp/steady-cpu-validation');args=parser.parse_args()
    validate(args.output)
