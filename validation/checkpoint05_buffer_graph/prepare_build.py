#!/usr/bin/env python3
"""Emit reviewed r5-derived freeze/build scripts. Does not run them."""
from pathlib import Path
import argparse,json,hashlib

def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--records',type=Path,required=True);a.add_argument('--template-records',type=Path,required=True);args=a.parse_args()
 args.records.mkdir(parents=True,exist_ok=True);emitted=[]
 for bits in [32,64]:
  old=args.template_records/f'build-scheme-r5-fp{bits}';new=args.records/f'build-fp{bits}';new.mkdir(exist_ok=False)
  for name in ['freeze-materialize.py','provenance.py','run_build.sh']:
   text=(old/name).read_text().replace(str(old),str(new)).replace('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-opt-checkpoint3-repair2-20261112',str(args.source))
   text=text.replace('repair2-accepted-build-r5','checkpoint5-buffer-graph').replace('validation/checkpoint03_combined','validation/checkpoint05_buffer_graph')
   text=text.replace('Unified constant/scheme controls and migrated inputs; compilation only, numerical acceptance not started.','Buffer capacity and precise graph reuse; compilation only, numerical checkpoint not started.')
   (new/name).write_text(text);emitted.append({'path':str(new/name),'sha256':hashlib.sha256(text.encode()).hexdigest()})
 driver=args.records/'build-eight.sh';driver.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+''.join('bash '+str(args.records/f'build-fp{bits}/run_build.sh')+'\n' for bits in [32,64]))
 (args.records/'build-preparation.json').write_text(json.dumps({'status':'SCRIPTS_ONLY_NOT_EXECUTED','source':str(args.source),'baseline':'3e93bcbe8c25e65765dbd1b9e154a73099ed6a26','files':emitted,'driver':str(driver)},indent=2))
if __name__=='__main__':main()
