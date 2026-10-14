import csv,json
from r2_runner import *
for name in ['SST-RK','moving-GCL','moving-chemistry']:
 folder=D/'artifacts/extra'/name
 if not folder.exists():continue
 print(name,'files',[q.name for q in folder.iterdir()])
 for p in folder.glob('*.csv'):
  if any(k in p.name for k in ['sst','summary','geometry','audit']):
   with p.open() as f:r=csv.reader(f);header=next(r);first=next(r,None);rows=list(r)
   print(p.name,'cols',header,'first',first,'last',rows[-1] if rows else None,'rows',len(rows)+1)
