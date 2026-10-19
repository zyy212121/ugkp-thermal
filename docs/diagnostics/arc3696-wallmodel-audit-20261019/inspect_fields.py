#!/usr/bin/env python3
"""Recompute selected-time field differences; numpy is the only dependency."""
from pathlib import Path
import argparse,re,json
import numpy as np
def field(path,n=None,vector=False):
 s=re.sub(r'/\*.*?\*/|//[^\n]*','',path.read_text(),flags=re.S);m=re.search(r'internalField\s+uniform\s+([^;]+);',s)
 if m:
  assert n is not None;v=np.fromstring(m.group(1).strip(' ()\n'),sep=' ');return np.tile(v,(n,1))if vector else np.full(n,v[0])
 kind='vector'if vector else'scalar';m=re.search(r'internalField\s+nonuniform\s+List<'+kind+r'>\s+(\d+)\s*\((.*?)\)\s*;',s,re.S);assert m,path
 values=np.fromstring(m.group(2).replace('(',' ').replace(')',' '),sep=' ');return values.reshape((-1,3))if vector else values
def timepath(case,wanted):
 for p in case.iterdir():
  if not p.is_dir():continue
  try:t=float(p.name)
  except ValueError:continue
  if abs(t-wanted)<1e-12:return p
 raise ValueError('Time not packaged: '+str(wanted))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--time',type=float,default=.018);a=p.parse_args();left=a.root/'baseline-boundaryLayer';right=a.root/'wallFunction-f9';l=timepath(left,a.time);r=timepath(right,a.time);vol=field(left/'0/V');rows=[]
 for name in ['U','T','p','rho','k','omega','nut','rhoE','rhoU']:
  x=field(l/name,len(vol),name in ['U','rhoU']);y=field(r/name,len(vol),name in ['U','rhoU']);d=y-x;err=np.linalg.norm(d,axis=1)if d.ndim==2 else abs(d);scale=np.linalg.norm(x,axis=1)if x.ndim==2 else abs(x)
  rows.append({'field':name,'RMS':float(np.sqrt(np.mean(err**2))),'volume_RMS':float(np.sqrt(np.average(err**2,weights=vol))),'max_abs':float(err.max()),'baseline_max_abs':float(scale.max()),'max_over_baseline_max':float(err.max()/scale.max())})
 print(json.dumps({'physical_time':a.time,'results':rows},indent=2))
if __name__=='__main__':main()
