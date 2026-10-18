import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,subprocess,hashlib
from pathlib import Path
import ctypes,csv,json,time
import numpy as np
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[4]);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=160);args=p.parse_args();root=args.source.resolve();D=args.output.resolve()
formal=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal').resolve()
if root==formal or D==formal or formal in D.parents:raise SystemExit('Use an independent TEST source/output')
D.mkdir(parents=True,exist_ok=False)
command=['g++','-std=c++17','-O2','-ffp-contract=off','-fPIC','-shared','-I'+str(root),str(Path(__file__).with_name('oracle.cpp')),'-o',str(D/'oracle.so')]
compiled=subprocess.run(command,capture_output=True,text=True);(D/'compile.log').write_text(compiled.stdout+compiled.stderr);compiled.check_returncode()
lib=ctypes.CDLL(str(D/'oracle.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.initialize.argtypes=[ptr,ptr,ptr];lib.evaluate.argtypes=[ptr,ptr]
inputs=json.loads(Path(__file__).with_name('inputs.json').read_text())['cases']
(D/'command.json').write_text(json.dumps({'compile':command,'compile_exit':compiled.returncode,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'budget':args.budget,'source_sha256':hashlib.sha256((root/'common/gasWall/ReactingWallLayer.H').read_bytes()).hexdigest(),'meaning':'Execution exit0 is not convergence. Inspect strict for every case; there is no production fallback or hand-written residual.'},indent=2))
results=[]
for entry in inputs:
 case=entry['case'];v=np.array(entry['values']);x=np.zeros(96);f=np.zeros(96);assert lib.initialize(v,x,f);initial=max(abs(f));np.savez(D/('case'+str(case)+'-initial.npz'),input=v,x=x,f=f)
 # Eliminate exactly prescribed boundary unknowns, and the uncoupled zero
 # second tangent / uniform frozen composition of this manufactured family.
 free=np.array([i*6+j for i in range(15)for j in [0,2,4,5]if i!=0 or j==5]);scale=np.ones(96)
 for i in range(16):scale[i*6]=v[4];scale[i*6+2]=max(v[2],v[3]);scale[i*6+4]=max(max(x[4::6]),v[5]);scale[i*6+5]=max(max(x[5::6]),1/np.sqrt(v[6]))
 lower=np.array([-v[2]/scale[k] if k%6==2 else 0 if k%6 in [4,5] else -np.inf for k in free]);upper=np.full(len(free),np.inf);base=x.copy();q=x[free]/scale[free];calls=0
 def residual(q):
  global calls
  calls+=1;full=base.copy();full[free]=q*scale[free];out=np.zeros(96)
  if not lib.evaluate(full,out):raise ArithmeticError('real operator rejected a trial')
  return out
 def jacobian(q):
  baseline=residual(q);matrix=np.empty((96,len(q)))
  for column in range(len(q)):
   h=np.cbrt(np.finfo(float).eps)*max(1,abs(q[column]));positive=q.copy();positive[column]+=h
   if q[column]-h>=lower[column]:
    negative=q.copy();negative[column]-=h;matrix[:,column]=(residual(positive)-residual(negative))/(positive[column]-negative[column])
   else:matrix[:,column]=(residual(positive)-baseline)/(positive[column]-q[column])
  return matrix
 begin=time.time()
 try:
  solved=least_squares(residual,q,method='trf',jac=jacobian,bounds=(lower,upper),x_scale='jac',diff_step=np.cbrt(np.finfo(float).eps),ftol=None,xtol=10*np.finfo(float).eps,gtol=10*np.finfo(float).eps,max_nfev=args.budget)
  final=base.copy();final[free]=solved.x*scale[free];ff=np.zeros(96);valid=bool(lib.evaluate(final,ff));np.savez(D/('case'+str(case)+'-final.npz'),x=final,f=ff,active_mask=solved.active_mask,gradient=solved.grad)
  item=dict(case=case,initial_residual=initial,final_residual=float(max(abs(ff))),strict=bool(valid and max(abs(ff))<=1.01e-8),valid=valid,nfev=solved.nfev,calls=calls,status=int(solved.status),message=str(solved.message),optimality=float(solved.optimality),active=int(np.count_nonzero(solved.active_mask)),elapsed=time.time()-begin)
 except Exception as e:item=dict(case=case,initial_residual=initial,calls=calls,error=str(e),elapsed=time.time()-begin)
 results.append(item);(D/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(item),flush=True)
