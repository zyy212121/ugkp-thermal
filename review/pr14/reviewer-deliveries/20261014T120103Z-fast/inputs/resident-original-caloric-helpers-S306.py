import pathlib,json,numpy as np,math,sys
from r3_runner import ROOT,D
sys.path.insert(0,str(ROOT/'common'))
from thermal_particle_restart_current import iter_chunks
import importlib.util
sp=importlib.util.spec_from_file_location('resident_probe',ROOT/'applications/gasUGKP/tools/ugkp_particle_restart.py');gas=importlib.util.module_from_spec(sp);sp.loader.exec_module(gas)
def particles(path):
 with path.open('rb') as f:head=f.readline().decode()
 chunks=[c for _,c in iter_chunks(path)] if head.startswith(('UGKP_CHT','UGKP_FSH')) else list(gas.iter_restart_chunks(path))
 return {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
def caloric(h):
 cp=lambda t:(153.5+0.0031*(min(max(t,2327.),4000.)-2327.))/.101961276
 hl=1273.*2317.+(.5*(1273.+cp(2327.))+1.16e6/20.)*20.
 if h<hl: return 2317.+(h-1273.*2317.)/(.5*(1273.+cp(2327.))+1.16e6/20.)
 dh=h-hl;s=.0031/.101961276;b=cp(2337.)
 return 2337.+2.*dh/(b+math.sqrt(b*b+2.*s*dh))
def finite_case(name):
 folder=D/'artifacts/resident'/name;states=[];bad=[]
 for q in sorted(folder.iterdir(),key=lambda p:float(p.name)):
  if not (q/'gpuResidentStrictParticles.dat').is_file():continue
  p=particles(q/'gpuResidentStrictParticles.dat');notfinite=[k for k,a in p.items() if np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all()]
  mismatch=None
  if 'coldWall' in name and not notfinite:
   nodes=p['pCold2DNodeSpecificEnthalpy' if '2D' in name else 'pColdNodeSpecificEnthalpy']
   carry=p['pThermalEnthalpyResidual'];mean=float(nodes.astype(np.float64).mean())+float(carry[0]);expected=caloric(mean);mismatch=abs(float(p['pT'][0])-expected)
   if mismatch>1e-9:bad.append({'time':q.name,'mean_profile_carry_temperature_abs':mismatch})
  if notfinite:bad.append({'time':q.name,'nonfinite_fields':notfinite})
  states.append({'time':q.name,'particle_count':len(p['pm']),'all_physical_particle_fields_finite':not notfinite,'physical_dtypes':{k:str(a.dtype) for k,a in p.items()},'profile_carry_temperature_abs':mismatch})
 report={'case':name,'states':states,'failures':bad,'scope':'All decoded actual particle fields and profile/carry caloric consistency, including written states. No global energy/wall ledger conservation claim.'}
 print(json.dumps(report,indent=2));raise SystemExit(bool(bad))
def restart(name):
 original=D/'artifacts/resident'/name/'1e-05';resumed=D/'artifacts/resident'/(name+'-RESUME')/'1e-05';a=particles(original/'gpuResidentStrictParticles.dat');b=particles(resumed/'gpuResidentStrictParticles.dat');rows=[];failed=[]
 for k in sorted(a):
  same=k in b and a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes();maximum=float(np.max(np.abs(a[k].astype(np.float64)-b[k].astype(np.float64)))) if k in b else None
  rows.append({'field':k,'bitwise_equal':same,'dtype':str(a[k].dtype),'shape':list(a[k].shape),'maximum_abs':maximum})
  if not same:failed.append(k)
 raw=[]
 for q in sorted(original.iterdir()):
  if not q.is_file():continue
  other=resumed/q.name;same=other.exists() and q.read_bytes()==other.read_bytes();raw.append({'path':q.name,'byte_equal':same})
  if not same:failed.append('written '+q.name)
 print(json.dumps({'case':name,'all_particle_members':rows,'all_terminal_files':raw,'failures':failed,'scope':'All final decoded particle members bitwise including IDs/RNG/8 or64 float nodes/double carry/history; all scalar/vector/hidden files byte equality at same final time. Directory uniform time metadata not compared.'},indent=2));raise SystemExit(bool(failed))
if __name__=='__main__':(restart if sys.argv[1]=='restart' else finite_case)(sys.argv[2])
