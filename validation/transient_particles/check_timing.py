#!/usr/bin/env python3
"""Isolated paired physical-step timings; the old transient path is unsupported.
The reference is PR28 statistical steady stepping with the same fixed dt and
static boundary. It also accumulates statistics, so this is not an old/new
transient speedup benchmark. No concurrent GPU tests should run.
"""
from pathlib import Path
import argparse,importlib.util,json,statistics
sp=importlib.util.spec_from_file_location('native',Path(__file__).with_name('check_native.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['candidate','reference','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--steps',type=int,default=6000);a=p.parse_args();o=a.output.resolve();o.mkdir(parents=True,exist_ok=False)
 result={'steps':a.steps,'dt':2e-5,'runs':[],'comparisons':[],'scope':__doc__,'binaries':{k:m.sha(getattr(a,k)) for k in ['candidate','reference']}}
 for rep in range(3):
  cases={}
  for key in (['reference','candidate'] if rep%2==0 else ['candidate','reference']):
   d=o/(key+'-'+str(rep));m.copy_inputs(m.ROOT/'validation/steady_cuda/evidence/cases/mixture_particles',d);m.control(d,a.steps,steady=key=='reference',write=a.steps);m.set_entry(d/'system/controlDict','windowSamples',a.steps)
   row=m.run(getattr(a,key).resolve(),d);row.update(binary=key,repetition=rep);result['runs'].append(row);assert row['exit']==0;cases[key]=d
   print(key,rep,row['seconds'],flush=True);(o/'results.json').write_text(json.dumps(result,indent=2)+'\n')
  aa=m.dirs(cases['reference'])[-1];bb=m.dirs(cases['candidate'])[-1];check=m.compare(aa,bb,32);check['repetition']=rep
  parcels=m.parcel(bb/'gpuResidentStrictParticles.dat');check['minimum_surviving_id']=min(x[-1] for x in parcels);check['maximum_surviving_id']=max(x[-1] for x in parcels)
  assert check['minimum_surviving_id']>1,'long run must exercise outflow'
  result['comparisons'].append(check)
 result['median_seconds']={k:statistics.median(x['seconds'] for x in result['runs'] if x['binary']==k) for k in ['candidate','reference']};result['pass']=True
 (o/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()