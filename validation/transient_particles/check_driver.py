#!/usr/bin/env python3
"""Transient driver regressions: old paths, physical schedules, adaptive dt."""
from pathlib import Path
import argparse,importlib.util,json,shutil
spec=importlib.util.spec_from_file_location('native',Path(__file__).with_name('check_native.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['candidate','before','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);report={'runs':[],'comparisons':[]}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def run(key,d):
  row=m.run(getattr(a,key).resolve(),d);row.update(binary=key,case=d.name);report['runs'].append(row);save();print(d.name,row['exit'],flush=True);assert row['exit']==0
 for fixture in ['reservoir','reconstructed','mixture_diffusion','particles']:
  cases=[]
  for key in ['before','candidate']:
   d=out/(fixture+'-'+key);m.copy_inputs(m.ROOT/'validation/steady_cuda/evidence/cases'/fixture,d);m.control(d,200,write=100);run(key,d);cases.append(d)
  aa=m.dirs(cases[0])[-1];bb=m.dirs(cases[1])[-1];errors={}
  for name in m.FIELD_NAMES:
   if not (aa/name).exists():continue
   x=m.fields.internal(aa/name,32);y=m.fields.internal(bb/name,32);err=max(abs(u-v)/(1+abs(u)) for xx,yy in zip(x,y) for u,v in zip(xx,yy));assert err<1e-8,(fixture,name,err);errors[name]=err
  if (aa/'gpuResidentStrictParticles.dat').exists():assert m.parcel(aa/'gpuResidentStrictParticles.dat')==m.parcel(bb/'gpuResidentStrictParticles.dat')
  report['comparisons'].append({'case':fixture,'max_scaled_errors':errors});save()
 # Adaptive Courant control constrained by the existing maximum physical dt.
 cases=[]
 for adaptive in [False,True]:
  d=out/('adaptive' if adaptive else 'fixed_dt');m.copy_inputs(m.ROOT/'validation/steady_cuda/evidence/cases/mixture_particles',d)
  m.control(d,100 if adaptive else 200,write=20,dt=2e-5 if adaptive else 1e-5)
  if adaptive:
   m.set_entry(d/'system/controlDict','adjustTimeStep','true')
   f=d/'system/controlDict';f.write_text(f.read_text()+'\nmaxDeltaT 1e-5; maxCo 100;\n')
  run('candidate',d);cases.append(d)
 row=m.compare(m.dirs(cases[0])[-1],m.dirs(cases[1])[-1],32);row['case']='adaptive vs fixed 1e-5 physical dt';report['comparisons'].append(row);assert len(m.dirs(cases[1]))==11;save()
 # Preserve the existing unsupported mixture scheduled-boundary contract.
 # This PR enables particle coupling, not a new mixture boundary operator.
 d=out/'scheduled_mixture_unsupported';m.copy_inputs(m.ROOT/'validation/steady_cuda/evidence/cases/mixture_particles',d);m.control(d,1,write=1)
 m.set_entry(d/'constant/schedulingProperties','gpuResidentDynamicInlet','true');m.set_entry(d/'constant/particleProperties','gpuResidentInletPatch','left')
 (d/'constant/inletPressure.table').write_text('((0 207861.56545383102) (0.004 212018.79676290765))\n')
 (d/'constant/inletVolumeFraction.table').write_text('((0 0.000001) (0.004 0.000002))\n')
 row=m.run(a.candidate.resolve(),d);row.update(binary='candidate',case=d.name,expected_exit=1);report['runs'].append(row)
 assert row['exit']!=0 and len(m.dirs(d))==1 and 'missing mixture boundary operator' in (d/'run.log').read_text()
 report['unchanged_limitation']='Scheduled pressure boundaries are not implemented for mixture gas; existing diagnostic retained.'
 report['pass']=True;save()
if __name__=='__main__':main()