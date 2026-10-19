#!/usr/bin/env python3
"""CPU execution of unchanged production operators on selected snapshots.
This is not CUDA replay and not a complete SST manufactured solution.
"""
from pathlib import Path
import argparse,subprocess,json,sys,hashlib
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);source=a.source.resolve();snap=a.snapshots.resolve();here=Path(__file__).resolve().parent
 assert(source/'common/operators/computeRiemannGasFaceFluxDevice.cuh').is_file()
 assert(snap/'baseline-boundaryLayer/constant/polyMesh/points').is_file()and(snap/'wallFunction-f9/0/U').is_file()
 execution=out/'execution';logs=out/'logs';execution.mkdir();logs.mkdir();cfg=out/'config.json';cfg.write_text(json.dumps({'source':str(source),'snapshots':str(snap),'execution':str(execution),'logs':str(logs)},indent=2));commands=[]
 def run(cmd):
  q=subprocess.run(cmd,cwd=source,text=True,capture_output=True);commands.append({'command':cmd,'exit':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2));print(q.stdout,end='');print(q.stderr,end='',file=sys.stderr)
  if q.returncode:raise SystemExit(q.returncode)
  return q
 for name in ['export_snapshot.py','build_budget.py','manufactured_shear.py','linear_controls.py']:run([sys.executable,str(here/name),str(cfg)])
 cmd=['g++','-std=c++17','-O2','-I'+str(source/'common'),str(here/'pr21-stationary-contact.cpp'),'-o',str(execution/'stationary-contact')];run(cmd);q=run([str(execution/'stationary-contact')]);(logs/'stationary-contact.csv').write_text(q.stdout);(logs/'stationary-contact.log').write_text(q.stderr)
 (out/'file-hashes.json').write_text(json.dumps({str(p.relative_to(logs)):hashlib.sha256(p.read_bytes()).hexdigest()for p in logs.rglob('*')if p.is_file()},indent=2));print('Completed CPU operator reproduction:',logs)
if __name__=='__main__':main()
