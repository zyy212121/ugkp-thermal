#!/usr/bin/env python3
"""Prepare local verification inputs, or explicitly invoke a future real adapter.

No CHMT solver or adapter is provided. No mock solver fallback exists.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import compare
import reference


def case_spec(case):
    matches = [c for c in reference.load_cases()['cases'] if c['id']==case]
    if not matches or matches[0]['reference_status']!='ANALYTIC_AVAILABLE':
        raise ValueError('unknown or DATA_REQUIRED case; complete external inputs first')
    return matches[0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(case, output):
    spec = case_spec(case)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be absent or empty; existing data will not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'case-spec.json',spec)
    reference.write_rows(output/'reference.csv',reference.generate(case))
    request = {'schema_version':1,'case':case,'status':'PREPARED_NOT_RUN',
               'adapter_status':'pendingCHMTadapter','solver_validation_status':'UNVERIFIED',
               'spec_sha256':digest(output/'case-spec.json'),
               'reference_sha256':digest(output/'reference.csv'),
               'required_outputs':['observations.csv','run.json','original solver inputs, meshes and logs'],
               'note':'Neutral mathematical specification; not an executable OpenFOAM case.'}
    write_json(output/'request.json',request)
    return request


def executable(path, label):
    path = Path(path).resolve()
    if not path.is_file() or not os.access(path,os.X_OK):
        raise ValueError(f'{label} must be an existing executable: {path}; no fallback is supplied')
    return path


def execute(case, output, adapter, solver):
    spec = case_spec(case)
    adapter,solver = executable(adapter,'adapter'),executable(solver,'solver')
    output = Path(output).resolve()
    if not (output/'request.json').is_file():
        raise ValueError('run prepare first; no implicit physical run or output overwrite')
    request = json.loads((output/'request.json').read_text(encoding='utf-8'))
    if request.get('case')!=case or request.get('status')!='PREPARED_NOT_RUN':
        raise ValueError('prepared request does not match this case')
    if digest(output/'case-spec.json')!=request['spec_sha256'] or digest(output/'reference.csv')!=request['reference_sha256']:
        raise ValueError('prepared inputs changed; create a fresh prepared directory')
    if json.loads((output/'case-spec.json').read_text(encoding='utf-8'))!=spec:
        raise ValueError('prepared specification differs from current frozen case')
    names = ('adapter.log','observations.csv','run.json','comparison.json','convergence.csv','convergence.json')
    if any((output/name).exists() for name in names):
        raise ValueError('prior run artifacts exist; prepare a new directory rather than overwrite')
    # Explicit local adapter contract, not an assumed OpenFOAM solver CLI.
    described = subprocess.run([str(adapter),'--describe'],capture_output=True,text=True,check=True,timeout=30)
    cap = json.loads(described.stdout)
    if cap.get('schema_version')!=1 or cap.get('artifact_kind')!='CHMT_ADAPTER' or case not in cap.get('supported_cases',[]):
        raise ValueError('adapter does not declare support for this specification/case')
    cmd = [str(adapter),'run','--spec',str(output/'case-spec.json'),'--solver',str(solver),'--output',str(output)]
    with (output/'adapter.log').open('w',encoding='utf-8') as log:
        done = subprocess.run(cmd,cwd=output,stdout=log,stderr=subprocess.STDOUT,check=False)
    if done.returncode:
        raise ValueError(f'adapter failed with exit {done.returncode}; inspect adapter.log; no validation claimed')
    if digest(output/'case-spec.json')!=request['spec_sha256'] or digest(output/'reference.csv')!=request['reference_sha256']:
        raise ValueError('adapter modified reference/specification; comparison refused')
    meta = json.loads((output/'run.json').read_text(encoding='utf-8'))
    compare.validate_run_metadata(meta)
    if meta['input_sha256']!=request['spec_sha256']:
        raise ValueError('run input_sha256 does not match prepared specification')
    actual = compare.read_csv(output/'observations.csv')
    if any(r.get('kind')!='SOLVER_OBSERVATION' for r in actual):
        raise ValueError('adapter output must be genuine labelled solver observations')
    result = compare.compare_rows(list(reference.generate(case)),actual)
    write_json(output/'comparison.json',result)
    checks = {'fields_and_balances':result['numeric_match']}
    if case in compare.ORDER_MIN:
        q = 'T_bar' if case=='stefan' else 'mass' if case=='ale_wave' else 'H_bar'
        rows = [{'case':case,'metric':q+'_L1','n':r['n'],'error':r['L1_scaled']} for r in result['norms'] if r['quantity']==q and r['time']==max(spec['times'])]
        with (output/'convergence.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=('case','metric','n','error'))
            writer.writeheader(); writer.writerows(rows)
        convergence = compare.check_convergence(rows)
        write_json(output/'convergence.json',convergence)
        checks['space_convergence'] = convergence['numeric_match']
    pending = ['independent provenance/build review','time-refinement study where applicable']
    if case in ('ale_free','ale_wave'):
        pending.append('complete stage GCL for every resolution: use compare.py gcl with actual dimensions')
    return {'solver_validation_status':'UNVERIFIED','numeric_checks':checks,'remaining_gates':pending,
            'note':'Only the explicitly requested local adapter was invoked; these checks are not a full-validation certificate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','run'):
        sub = modes.add_parser(mode)
        sub.add_argument('--case',required=True)
        sub.add_argument('--output',required=True,type=Path)
        if mode=='run':
            sub.add_argument('--adapter',required=True,type=Path)
            sub.add_argument('--solver',required=True,type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.case,args.output) if args.mode=='prepare' else execute(args.case,args.output,args.adapter,args.solver)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if args.mode=='prepare' or all(result['numeric_checks'].values()) else 1
    except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
        print(json.dumps({'error':str(exc),'solver_validation_status':'UNVERIFIED'},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
