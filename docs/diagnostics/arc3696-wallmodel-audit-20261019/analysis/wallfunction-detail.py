exec(open('/mnt/c/Users/25094/Documents/Codex/2026-10-16/1-uniform-work-accepted-source-manifest/work/wallfunction-diagnose.py').read().split('times=[.0005')[0])
import subprocess,shutil
V=m.read_field(B/'0/V',n)
files={case:{f.relative_to(case).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()for d in ['0','constant','system']for f in(case/d).rglob('*')if f.is_file()}for case in [A,B]}
inputrows=[{'path':k,'same':files[A].get(k)==files[B].get(k),'baseline_sha256':files[A].get(k),'wallfunction_sha256':files[B].get(k)}for k in sorted(files[A].keys()|files[B].keys())]
m.write_csv(O/'input-comparison.csv',inputrows)
metrics=[];cross=[];replay=[];perstations=[]
for t in [.0005,.004,.018]:
 data={}
 for label,case in [('boundaryLayer-old',A),('wallFunction-current',B)]:
  tp=m.numeric_dirs(case)[round(t,12)];data[label]={name:m.read_field(tp/name,n,name in ['U','rhoU'])for name in ['U','T','p','rho','k','omega','nut','rhoE','rhoU']}
 for name,a in data['boundaryLayer-old'].items():
  b=data['wallFunction-current'][name];diff=b-a;err=np.linalg.norm(diff,axis=1)if diff.ndim==2 else abs(diff);norm=np.linalg.norm(a,axis=1)if a.ndim==2 else abs(a)
  mask=(C[:,1]<.002);near=err[mask];far=err[~mask]
  metrics.append({'time':t,'field':name,'RMS':float(np.sqrt(np.mean(err**2))),'volume_RMS':float(np.sqrt(np.average(err**2,weights=V))),'max_abs':float(err.max()),'baseline_max_abs':float(norm.max()),'max_relative_to_baseline_max':float(err.max()/norm.max()),'argmax_cell':int(err.argmax()),'argmax_x':float(C[err.argmax(),0]),'argmax_y':float(C[err.argmax(),1]),'nearwall_RMS_y_lt_2mm':float(np.sqrt(np.mean(near**2))),'far_RMS_y_ge_2mm':float(np.sqrt(np.mean(far**2)))})
 if t!=.018:continue
 oldflux={int(r['face']):r for r in csv.DictReader((m.numeric_dirs(A)[t]/'uniform/gasBoundaryLayer.csv').open())}
 profiles={}
 for r in csv.DictReader((m.numeric_dirs(A)[t]/'uniform/gasBoundaryLayerProfiles.csv').open()):profiles.setdefault(int(r['face']),[]).append(r)
 for label,d in data.items():
  for x in [.35,.38,.58,.75,.85]:
   j=int(np.argmin(abs(C[owners,0]-x)));c=int(owners[j]);face=int(faceids[j]);fl=oldflux[face];prof=sorted(profiles[face],key=lambda r:int(r['node']));edge=prof[-1];L=float(fl['matchingDistance']);distance=np.linalg.norm(C-np.array([C[c,0],L,.0005]),axis=1);donor=int(distance.argmin());assert distance[donor]<1e-10
   vals=[label,face,d['p'][donor],290.53216418474562,L,y[j],d['T'][donor],d['k'][donor],d['omega'][donor],*d['U'][donor],1,0,d['T'][c],*d['U'][c],d['k'][c],d['omega'][c],d['rho'][c],d['p'][c]]
   replay.append(vals)
   for k in range(n):
    if abs(C[k,0]-C[c,0])<1e-10:cross.append({'version':label,'x_station':x,'cell':k,'x':C[k,0],'y':C[k,1],**{key:float(v[k,0]if v.ndim==2 else v[k])for key,v in d.items()}})
   if label=='boundaryLayer-old':
    for r in prof:r=dict(r);r['x_station']=x;perstations.append(r)
m.write_csv(O/'field-differences.csv',metrics);m.write_csv(O/'five-station-profiles.csv',cross);m.write_csv(O/'baseline-internal-profiles.csv',perstations)
with(O/'same-input-wall-models.csv').open('w')as f:
 w=csv.writer(f);w.writerow(['dataset','face','pressure','Tw','L','y','edgeT','edgek','edgeomega','edgeUx','edgeUy','edgeUz','YAir','YCO2','ownerT','ownerUx','ownerUy','ownerUz','ownerk','owneromega','ownerrho','ownerp']);w.writerows(replay)
oldheader=subprocess.check_output(['git','show','7c696b19:common/gasNumerics/OpenFoamWallFunctions.cuh'],cwd=S,text=True);(O/'old-OpenFoamWallFunctions.cuh').write_text(oldheader)
print('FINAL FIELD DIFFERENCES',json.dumps([r for r in metrics if r['time']==.018],indent=2));print('DONORS',[(r[0],r[1],r[6],r[7],r[8])for r in replay]);print('INPUT COUNT',len(inputrows),'different',[r['path']for r in inputrows if not r['same']])
