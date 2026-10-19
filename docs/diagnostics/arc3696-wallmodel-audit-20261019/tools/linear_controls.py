import sys,json
C=json.loads(open(sys.argv[1]).read())
from pathlib import Path
import json,shutil,subprocess,csv
import numpy as np
R=Path(C['source']);E=Path(C['execution']);O=Path(C['logs']);source=E/'manufactured-shear/yp80-n16';definition=json.loads((source/'definition.json').read_text());N=16;H=definition['domain_height'];y=np.fromfile(source/'Cy.bin','<f8');own=np.fromfile(source/'faceOwner.bin','<i4');counts=(source/'counts.txt').read_text().split();ni=int(counts[2]);results=[]
for sign in [1,-1]:
 root=E/('linear-shear-'+str(sign));shutil.copytree(source,root,dirs_exist_ok=True);out=O/root.name;out.mkdir(exist_ok=True);U=sign*20*y/H;nuEff=H/20
 for name,value in {'Ux':U,'rhoUx':U,'rhoE':np.full(N,(1004.675-287.0500676)*300)+.5*U*U,'nut':np.full(N,nuEff-1.7244421537657099e-5)}.items():value.astype('<f8').tofile(root/(name+'.bin'))
 for prefix in ['gasBoundary','riemannBoundary']:
  vals=U[own].copy();vals[ni]=0;vals[ni+1]=sign*20;vals.astype('<f8').tofile(root/(prefix+'Ux.bin'))
 args=[str(E/'snapshot-budget'),str(root),str(out),'1'];r=subprocess.run(args,capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 rows=list(csv.DictReader((out/'normal-face-budget.csv').open()));err=max(abs(float(t['full_mx'])+sign)for t in rows);num=max(abs(float(t['numerical_mx']))for t in rows);assert err<1e-9 and num<1e-9
 results.append({'sign':sign,'cells':N,'expected_signed_face_momentum_flux':-sign,'max_total_flux_error':err,'max_numerical_flux':num,'command':args,'run_exit':r.returncode});print(results[-1])
(O/'linear-control-summary.json').write_text(json.dumps(results,indent=2))
dest=O/'unsupported-mixture-HLLC';dest.mkdir(exist_ok=True);args=[str(E/'snapshot-budget'),str(E/'wallfunction-018'),str(dest),'1','unsupported-dispatch'];r=subprocess.run(args,capture_output=True,text=True);(dest/'run.log').write_text(r.stdout+r.stderr);(dest/'command.json').write_text(json.dumps({'command':args,'exit':r.returncode,'meaning':'Expected MissingOperator diagnostic; NOT a working mixture HLLC path'},indent=2));assert r.returncode==0;print(r.stdout)
