from pathlib import Path
import numpy as np,sys,re,json,csv
R=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr21-wallmodel-audit-20261019')
sys.path.insert(0,str(R/'docs/diagnostics/arc3696-wallmodel-audit-20261019/tools'))
from foam_io import read_field,numeric_dirs,strip_comments
B=R.parent/'pr21-audit-execution-20261019'
O=Path('/mnt/d/tempt/pr21-wallmodel-audit-20261019/of10-reference')
S=R/'isolated-extract-validation'
cases={'baseline':S/'baseline-boundaryLayer','wallFunction':S/'wallFunction-f9'}
for ny in map(int,sys.argv[1:]or['80','160']):
 cases[f'OF10-{ny}']=B/f'of10-reference-{ny}'
 work=B/f'of10-reference-work-{ny}'
 if work.exists()and max(numeric_dirs(work))>0:cases[f'OF10-work-{ny}']=work
stations=[.35,.38,.58,.75,.85];rows=[];prof=[];globalcomp=[]
def patchvalue(p,patch,n,vector=False):
 s=strip_comments(p.read_text());body=re.search(r'\b'+patch+r'\s*\{([^}]+)\}',s,re.S).group(1)
 a=re.search(r'value\s+nonuniform\s+List<\w+>\s+\d+\s*\((.*?)\)\s*;',body,re.S)
 if a:
  v=np.fromstring(a[1].replace('(',' ').replace(')',' '),sep=' ');return v.reshape((-1,3))if vector else v
 a=re.search(r'value\s+uniform\s+([^;]+)',body);v=np.fromstring(a[1].strip(' ()\n'),sep=' ');return np.tile(v,(n,1))if vector else np.full(n,v[0])
allfields={}
for label,case in cases.items():
 ds=numeric_dirs(case);d=ds[max(ds)]if label.startswith('OF10') else min(ds.values(),key=lambda p:abs(float(p.name)-.018))
 n=int(re.search(r'nonuniform\s+List<vector>\s+(\d+)',(d/'U').read_text())[1]);C=read_field((d/'C')if label.startswith('OF10')else case/'0/C',n,True)
 f={name:read_field(d/name,n,name=='U')for name in ['U','T','p','k','omega','nut']};allfields[label]=(C,f)
 walltau=None
 if label.startswith('OF10'):
  walltau=np.concatenate([patchvalue(d/'wallShearStress',p,nw,True)for p,nw in [('upstreamWall',96),('porousWall',192),('downstreamWall',48)]])
  yp=np.concatenate([patchvalue(d/'yPlus',p,nw)for p,nw in [('upstreamWall',96),('porousWall',192),('downstreamWall',48)]])
 xgrid=np.unique(np.round(C[:,0],11));ygrid=np.unique(np.round(C[:,1],11));assert len(xgrid)*len(ygrid)==n,(label,len(xgrid),len(ygrid),n)
 for station in stations:
  x=min(xgrid,key=lambda v:abs(v-station));ids=np.flatnonzero(np.abs(C[:,0]-x)<1e-8);ids=ids[np.argsort(C[ids,1])];i=int(ids[0]);ix=int(np.argmin(abs(xgrid-x)))
  row={'case':label,'iteration_or_time':d.name,'station':station,'x':float(x),'owner':i,'y':float(C[i,1]),**{name:float(f[name][i,0]if name=='U'else f[name][i])for name in f}}
  if walltau is not None:row.update(wall_tau_x=float(walltau[ix,0]),yplus=float(yp[ix]))
  rows.append(row)
  for j in ids:
   prof.append({'case':label,'station':station,'x':float(x),'y':float(C[j,1]),'U':float(f['U'][j,0]),'k':float(f['k'][j]),'T':float(f['T'][j]),'omega':float(f['omega'][j]),'nut':float(f['nut'][j])})
for label,(C,f)in allfields.items():
 if label.startswith('OF10')and len(C)==26880:
  # Mesh owner numbering should be identical after blockMesh; assert centers.
  for name,(C2,f2)in allfields.items():
   if name in ['baseline','wallFunction']:
    assert np.max(abs(C-C2))<1e-10
    globalcomp.append({'reference':label,'comparison':name,**{k+'_RMS':float(np.sqrt(np.mean(np.sum((f2[k]-f[k])**2,axis=1)if k=='U'else(f2[k]-f[k])**2)))for k in f}})
common=[]
for r in rows:
 if not r['case'].startswith('OF10'):continue
 for target in ['baseline','wallFunction']:
  tr=next(s for s in rows if s['case']==target and s['station']==r['station'])
  ps=[p for p in prof if p['case']==r['case']and p['station']==r['station']]
  common.append({'reference':r['case'],'target':target,'station':r['station'],'y':tr['y'],**{k:float(np.interp(tr['y'],[p['y']for p in ps],[p[k]for p in ps]))for k in ['U','k','T','omega','nut']}})
for name,data in [('stations.csv',rows),('profiles.csv',prof)]:
 with(O/name).open('w',newline='')as h:
  keys=list(dict.fromkeys(k for r in data for k in r));w=csv.DictWriter(h,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(data)
(O/'comparison.json').write_text(json.dumps({'stations':rows,'same_mesh_RMS':globalcomp,'reference_interpolated_at_original_first_center':common},indent=2))
for r in rows:print(r)
print(globalcomp)
