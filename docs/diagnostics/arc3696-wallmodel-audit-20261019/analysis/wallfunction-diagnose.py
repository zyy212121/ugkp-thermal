from pathlib import Path
import re,json,csv,hashlib,datetime,importlib.util
import numpy as np
from zoneinfo import ZoneInfo
S=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018')
A=Path('/home/lss/ARC3696-runs/20261017T042008+0800/M0p55/baseline-rh1au35z')
B=Path('/home/lss/ARC3696-runs/pr19-f9e8d775-20261019T185625+0800/M0p55/baseline-lae7a4r6')
O=Path('/mnt/d/tempt/wallfunction-vs-boundarylayer-20261019');O.mkdir(exist_ok=True)
helper=Path('/mnt/c/Users/25094/Documents/Codex/2026-10-16/files-pasted-by-the-user-wsl/ARC3696-baseline-comparison/compare_arc3696_baseline.py')
spec=importlib.util.spec_from_file_location('compare_helpers',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
C=m.read_field(B/'0/C',vector=True);n=len(C)
owner=np.array([int(x)for x in m.list_body((B/'constant/polyMesh/owner').read_text()).split()])
faceids=np.arange(53504,53840);owners=owner[faceids];y=C[owners,1]
times=[.0005,.004,.018];rows=[]
for t in times:
 for label,case in [('boundaryLayer-old',A),('wallFunction-current',B)]:
  tp=m.numeric_dirs(case)[round(t,12)];f={name:m.read_field(tp/name,n,name=='U')for name in ['U','T','rho','p','k','omega','nut']}
  rhoW=f['p'][owners]/(287.0500676*290.53216418474562)
  us=.09**.25*np.sqrt(f['k'][owners]);yp=us*y*rhoW/1.7244421537657099e-5
  print('WALL',t,label,'y range',y.min(),y.max(),'yPlus_k',np.percentile(yp,[0,25,50,75,100]).tolist(),'k',np.percentile(f['k'][owners],[0,50,100]).tolist())
  for x in [.35,.38,.58,.75,.85]:
   j=int(np.argmin(abs(C[owners,0]-x)));c=int(owners[j]);row={'time':t,'version':label,'face':int(faceids[j]),'cell':c,'x':C[c,0],'y':y[j],'uStar_k':us[j],'yPlus_k':yp[j]}
   for name in f:row[name]=float(f[name][c,0]if name=='U'else f[name][c])
   rows.append(row);print(row)
  if label=='wallFunction-current':
   with(O/f'wall-input-t{t}.csv').open('w')as out:
    w=csv.writer(out);w.writerow(['face','cell','x','y','U','T','p','rhoWall','k','omega','nut'])
    for i,c in enumerate(owners):w.writerow([faceids[i],c,C[c,0],y[i],np.linalg.norm(f['U'][c]),f['T'][c],f['p'][c],rhoW[i],f['k'][c],f['omega'][c],f['nut'][c]])
  if t==.018:
   print('UNIFORM',label,[p.name for p in(tp/'uniform').iterdir()])
with(O/'five-station-owner-values.csv').open('w')as out:w=csv.DictWriter(out,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(O/'initial-observation.json').write_text(json.dumps({'current_case':str(B),'reference_case':str(A),'source_sha':'f9e8d77559c9d5f0eb884755203c22d2606aae7e','current_wall_model':'wallFunction','reference_wall_model':'boundaryLayer/reactingSst','input_changes':['constant/fluidProperties wall model','system/controlDict endTime'],'cell_count':n,'station_rows':rows},indent=2))
