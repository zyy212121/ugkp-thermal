#!/usr/bin/env python3
"""Compile/run independent real-header lineage evidence, never editing source-root."""
from pathlib import Path
import argparse,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--source-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--label',required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
headers=output/'instrumented';subprocess.run([sys.executable,str(here/'instrument.py'),str(a.source_root.resolve()),str(headers)],check=True)
cases=['ordinary','full_success_half_failure','full_failure_two_half_success','first_half_only_then_remainder','full_success_second_half_failure','newton_zero','linesearch_zero','both_zero','refinement_one','ad_only_failure','singular_jacobian','failed_armijo','inversion_local_stop','step_budget_zero','remainder_budget','persistent_lte','undefined_lte','below_minimum','subnormal','subnormal_2','subnormal_3','subnormal_4','subnormal_5','subnormal_7','subnormal_17','subnormal_1023','absorbed_update','equilibrium','zero_interval','unpaid_tail','tiny_unpaid_tail','absorbed_remaining']
manifest={'label':a.label,'source':str(a.source_root.resolve()),'builds':[],'cases':[]}
for name in ['lineage','wider_time','repeated_call','multiple_trial']:
 command=['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(headers),str(here/(name+'_probe.cpp')),'-o',str(output/name)]
 r=subprocess.run(command,capture_output=True,text=True);(output/(name+'_compile.log')).write_text(r.stdout+r.stderr);manifest['builds'].append({'command':command,'returncode':r.returncode});r.check_returncode()
for name,args in [('lineage',[case]) for case in cases]+[('wider_time',[]),('wider_time',['1e-44']),('repeated_call',[]),('multiple_trial',[])]:
 case=args[0] if name=='lineage' else name+('_'+args[0] if args else '')
 command=[str(output/name),*args]
 try:r=subprocess.run(command,capture_output=True,text=True,timeout=15);code=r.returncode;log=r.stdout+r.stderr
 except subprocess.TimeoutExpired as e:code=124;log=str(e)
 (output/(case+'.txt')).write_text(log);manifest['cases'].append({'case':case,'command':command,'returncode':code});print(case,code)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sys.exit(any(x['returncode'] for x in manifest['cases']))
