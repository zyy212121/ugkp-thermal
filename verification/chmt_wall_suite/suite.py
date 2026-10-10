#!/usr/bin/env python3
"""Prepare/drive all requested case families without conflating backend evidence."""
import argparse
import json
from pathlib import Path
from catalog import cases
from generate import prepare
from run import execute

def prepare_suite(root,names,options):
    root=Path(root)
    if root.exists():raise FileExistsError('suite output must be fresh')
    root.mkdir(parents=True);catalog=cases();summary={'cases':[],'unsupported':[],'native_acceptance':'NOT_RUN'}
    for name in names:
        spec=catalog[name]
        for family in spec['wall_variants'] or [None]:
            opts=dict(options.get('common',{}));opts.update(options.get(name,{}));opts.update(options.get('wall_families',{}).get(family,{}))
            if family:opts['wall']=family
            suffix=name+('__'+family if family else '');path=root/suffix
            manifest=prepare(name,path,opts);summary['cases'].append({'id':suffix,'path':str(path.resolve()),'category':manifest['category'],'gate':manifest['gate'],'solver':manifest['solver'],'species_count':manifest['species_count'],'family':family,'native_execution':'NOT_RUN'})
        if spec['solver']=='CHMT' and spec['category']==3:
            summary['unsupported'].append({'case':name,'family':'wallFunction','status':'UNSUPPORTED','reason':'Ordinary wallFunction is unsupported in CHMT; no fallback substitution.'})
    (root/'suite_manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');return summary

def select_solver(item,chmt,gas2,gas10):
    if item['solver']=='CHMT':return chmt
    if item['species_count']==2:return gas2
    if item['species_count']==10:return gas10
    raise ValueError('no binary selector for requested gas species count')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',choices=sorted(cases()),default=list(cases()));p.add_argument('--options',type=Path);p.add_argument('--mode',choices=['prepare','mesh','check-input','execute'],default='prepare');p.add_argument('--required',action='store_true');p.add_argument('--chmt-input-checker');p.add_argument('--gas-input-checker2');p.add_argument('--gas-input-checker10');p.add_argument('--chmt-solver');p.add_argument('--gas-solver','--gas-solver2',dest='gas_solver2');p.add_argument('--gas-solver10');a=p.parse_args()
    summary=prepare_suite(a.output,a.cases,json.loads(a.options.read_text()) if a.options else {})
    if a.mode!='prepare':
        for item in summary['cases']:
            checker=a.chmt_input_checker if item['solver']=='CHMT' else (a.gas_input_checker10 if item['species_count']==10 else a.gas_input_checker2)
            solver=select_solver(item,a.chmt_solver,a.gas_solver2,a.gas_solver10)
            item['result']=execute(item['path'],a.mode,solver,checker,allow_solver_fallback=False)
        summary['counts']={key:sum(i['result']['outcome']==key for i in summary['cases']) for key in ('PASS','FAIL','NOT_RUN','REPORT_ONLY')}
        summary['native_acceptance']='FAIL' if summary['counts']['FAIL'] else 'NOT_RUN' if summary['counts']['NOT_RUN'] else 'COMPLETED_WITH_REPORT_ONLY_CASES'
    (a.output/'suite_manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
    if any(i.get('result',{}).get('outcome')=='FAIL' for i in summary['cases']):raise SystemExit(1)
    if a.mode=='execute' and any(i['result']['native_execution']=='NOT_RUN' for i in summary['cases']):raise SystemExit(3)
    if a.required and a.mode!='prepare':
        requested={'mesh':'mesh','check-input':'input_import','execute':'native_execution'}[a.mode]
        if any(i['result'][requested] in ('NOT_RUN','UNSUPPORTED') for i in summary['cases']):raise SystemExit(3)
if __name__=='__main__':main()
