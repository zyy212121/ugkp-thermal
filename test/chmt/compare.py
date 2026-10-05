#!/usr/bin/env python3
"""Compare exported observations; numeric matches never certify solver validation."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import reference

EPS = sys.float_info.epsilon
LOCAL_TOL = 256*EPS
ACCUMULATED_TOL = 5e-10
ORDER_MIN = {'stefan':0.9,'film_plug':1.8,'film_shear':1.8,'ale_wave':1.8}
REQUIRED_METADATA = ('solver_commit','adapter_commit','build_id','input_sha256','precision','hardware','compiler','command','started_utc','time_step','steps','source_output_sha256')


def finite(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')
    return x


def validate_run_metadata(data):
    if data.get('artifact_kind') != 'SOLVER_OBSERVATION':
        raise ValueError('run metadata must identify genuine SOLVER_OBSERVATION, not analytic/self-test data')
    missing = [k for k in REQUIRED_METADATA if k not in data or data[k] in ('',None)]
    if missing:
        raise ValueError('missing run metadata: '+', '.join(missing))
    if data['precision'] != 'FP64':
        raise ValueError('these criteria are frozen for FP64 only')
    if not isinstance(data['steps'],int) or data['steps'] < 0:
        raise ValueError('steps must be a nonnegative integer')
    if finite(data['time_step'],'time_step') < 0:
        raise ValueError('time_step must be nonnegative')
    for key in ('input_sha256','source_output_sha256'):
        val = data[key]
        if not isinstance(val,str) or len(val)!=64 or any(c not in '0123456789abcdef' for c in val):
            raise ValueError(f'{key} must be a lowercase SHA-256')


def index_rows(rows):
    result = {}
    for r in rows:
        key = reference.row_key(r)
        finite(key[3],'time')
        value = finite(r['value'],'value')
        if key in result:
            raise ValueError(f'duplicate observation: {key}')
        result[key] = (value,r)
    if not result:
        raise ValueError('empty observations')
    return result


def tolerance(case, quantity, n, t):
    if t == 0:
        return LOCAL_TOL,LOCAL_TOL
    if case == 'stefan' and quantity in ('T_bar','h_bar','front','wall_heat'):
        factor = 128/n
        return (0.01*factor,0.05*factor) if quantity in ('T_bar','h_bar') else (0.01*factor,0.01*factor)
    if case in ('film_plug','film_shear') and quantity == 'H_bar':
        factor = (128/n)**2
        return 0.005*factor,0.02*factor
    if case == 'ale_wave' and quantity != 'volume':
        factor = (32/n)**2
        return 0.03*factor,0.08*factor
    if case == 'ale_free' and quantity != 'volume':
        return ACCUMULATED_TOL,ACCUMULATED_TOL
    return LOCAL_TOL,LOCAL_TOL


def compare_rows(expected, actual):
    exp,act = index_rows(expected),index_rows(actual)
    if exp.keys() != act.keys():
        raise ValueError(f'row keys differ: missing={len(exp.keys()-act.keys())}, extra={len(act.keys()-exp.keys())}')
    grouped = defaultdict(list)
    for key,(truth,r) in exp.items():
        scale = finite(r['scale'],'reference scale')
        if scale <= 0:
            raise ValueError('reference scale must be positive')
        got = act[key][0]
        # For integral quantities, scale includes cell volume: weights yield physical L1.
        error = abs(got-truth)/scale
        group = key[0:4]+(key[5],)
        grouped[group].append((error,scale))
    reports = []
    match = True
    for group,vals in sorted(grouped.items()):
        cid,variant,n,t,quantity = group
        L1 = math.fsum(e*s for e,s in vals)/math.fsum(s for e,s in vals)
        Linf = max(e for e,s in vals)
        tol1,tolinf = tolerance(cid,quantity,n,t)
        ok = L1 <= tol1 and Linf <= tolinf
        match &= ok
        reports.append(dict(case=cid,variant=variant,n=n,time=t,quantity=quantity,L1_scaled=L1,Linf_scaled=Linf,L1_limit=tol1,Linf_limit=tolinf,numeric_match=ok))
    balances = check_balances(exp,act)
    match &= all(x['numeric_match'] for x in balances)
    return {'numeric_match':bool(match),'solver_validation_status':'UNVERIFIED',
            'scope':'Field/balance comparison only. Separate convergence, stage GCL, provenance review and actual solver execution remain required.',
            'norms':reports,'balances':balances}


def check_balances(exp,act):
    groups = defaultdict(list)
    for key,(value,r) in act.items():
        groups[key[:4]].append((key,value))
    out = []
    for (cid,variant,n,t),rows in sorted(groups.items()):
        quantities = defaultdict(list)
        for key,value in rows:
            quantities[key[5]].append((key,value))
        if cid in ('ale_free','ale_wave','remap'):
            for quantity,values in quantities.items():
                expected_sum = math.fsum(exp[key][0] for key,val in values)
                actual_sum = math.fsum(val for key,val in values)
                scale = math.fsum(abs(exp[key][0]) for key,val in values)
                err = abs(actual_sum-expected_sum)/scale
                limit = 256*EPS*max(n,1) if cid=='remap' or quantity=='volume' else ACCUMULATED_TOL
                out.append(dict(case=cid,variant=variant,n=n,time=t,quantity=quantity+'_global',scaled_residual=err,limit=limit,numeric_match=err<=limit))
        elif cid in ('stefan','film_plug','film_shear'):
            quantity = 'h_bar' if cid=='stefan' else 'H_bar'
            energy = math.fsum(val/n for key,val in quantities[quantity])
            initial = math.fsum(value/n for key,(value,r) in act.items() if key[0:3]==(cid,variant,n) and key[3]==0 and key[5]==quantity)
            boundary_q = 'wall_heat' if cid=='stefan' else 'boundary_work'
            boundary = quantities[boundary_q][0][1]
            scale = max(abs(initial),abs(energy),abs(boundary))
            err = abs(energy-initial-boundary)/scale
            out.append(dict(case=cid,variant=variant,n=n,time=t,quantity='energy_budget',scaled_residual=err,limit=ACCUMULATED_TOL,numeric_match=err<=ACCUMULATED_TOL))
    return out


def check_convergence(rows):
    groups = defaultdict(list)
    for r in rows:
        cid = r['case']
        if cid not in ORDER_MIN:
            raise ValueError('no frozen convergence order for '+cid)
        n,e = int(r['n']),finite(r['error'],'error')
        if n <= 0 or e < 0:
            raise ValueError('positive n and nonnegative errors required')
        groups[(cid,r['metric'])].append((n,e))
    if not groups:
        raise ValueError('empty convergence data')
    out = []
    for (cid,metric),vals in sorted(groups.items()):
        vals.sort()
        expected_n = next(c['n'] for c in reference.load_cases()['cases'] if c['id']==cid)
        if [n for n,e in vals] != expected_n:
            raise ValueError(f'{cid}: require exactly the frozen three resolutions {expected_n}')
        for (n1,e1),(n2,e2) in zip(vals,vals[1:]):
            order = math.log(e1/e2)/math.log(n2/n1) if e1>LOCAL_TOL and e2>LOCAL_TOL else None
            ok = order is not None and math.isfinite(order) and order >= ORDER_MIN[cid]
            out.append(dict(case=cid,metric=metric,n_coarse=n1,n_fine=n2,order=order,minimum=ORDER_MIN[cid],numeric_match=ok))
    return {'numeric_match':all(x['numeric_match'] for x in out),'orders':out,'solver_validation_status':'UNVERIFIED',
            'note':'Zero/roundoff-dominated errors do not establish convergence; report an inconclusive study rather than a fabricated order.'}


def check_gcl(rows, expected_shape=None):
    seen = set()
    worst = 0.0
    for r in rows:
        key = tuple(int(r[k]) for k in ('step','stage','cell'))
        if key in seen:
            raise ValueError('duplicate stage/cell GCL record')
        seen.add(key)
        old,new = finite(r['V_old'],'V_old'),finite(r['V_new'],'V_new')
        if min(old,new) <= 0:
            raise ValueError('nonpositive volume')
        sweeps = [finite(r[k],k) for k in ('sweep_left','sweep_right','sweep_bottom','sweep_top')]
        residual = new-old-math.fsum(sweeps)
        scale = max(old,new,math.fsum(abs(s) for s in sweeps))
        worst = max(worst,abs(residual)/scale)
    if not seen:
        raise ValueError('empty GCL ledger')
    if expected_shape:
        steps,stages,cells = expected_shape
        required = {(s,k,c) for s in range(steps) for k in range(stages) for c in range(cells)}
        if seen != required:
            raise ValueError('incomplete or unexpected stage ledger; zero-based step/stage/cell indices required')
    return {'numeric_match':worst<=64*EPS,'max_scaled_residual':worst,'limit':64*EPS,
            'rows':len(seen),'solver_validation_status':'UNVERIFIED'}


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    fields = modes.add_parser('fields')
    fields.add_argument('--observations',required=True,type=Path)
    fields.add_argument('--run',required=True,type=Path)
    fields.add_argument('--case',required=True)
    conv = modes.add_parser('convergence')
    conv.add_argument('--input',required=True,type=Path)
    gcl = modes.add_parser('gcl')
    gcl.add_argument('--input',required=True,type=Path)
    for flag in ('steps','stages','cells'):
        gcl.add_argument('--'+flag,required=True,type=int)
    args = parser.parse_args()
    try:
        if args.mode=='fields':
            metadata = json.loads(args.run.read_text(encoding='utf-8'))
            validate_run_metadata(metadata)
            actual = read_csv(args.observations)
            if any(r.get('kind')!='SOLVER_OBSERVATION' for r in actual):
                raise ValueError('every observation row must identify SOLVER_OBSERVATION; reference CSV is not a run')
            result = compare_rows(list(reference.generate(args.case)),actual)
        elif args.mode=='convergence':
            result = check_convergence(read_csv(args.input))
        else:
            if min(args.steps,args.stages,args.cells)<=0:
                raise ValueError('positive ledger dimensions required')
            result = check_gcl(read_csv(args.input),(args.steps,args.stages,args.cells))
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['numeric_match'] else 1
    except (ValueError,KeyError,TypeError,OverflowError,OSError) as exc:
        print(json.dumps({'error':str(exc),'solver_validation_status':'UNVERIFIED'}),file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
