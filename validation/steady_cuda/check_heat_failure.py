#!/usr/bin/env python3
"""Native PR28-P2-001 regression. Source OpenFOAM bashrc before running.

Generate fresh one-cell gas/particle restart cases, then use the actual solver.
--expect-before documents the old false-success behavior (no changed assertions
inside production code). Output retains inputs, logs, exits and binary hashes.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,re,shutil,struct,subprocess

ROOT=Path(__file__).resolve().parents[2]

def prepare(d,mass,diameter,end_time,window):
 shutil.copytree(ROOT/'validation/steady_cuda/evidence/cases/mixture_particles',d,
                 ignore=shutil.ignore_patterns('polyMesh'))
 f=d/'system/blockMeshDict';f.write_text(f.read_text().replace('2.0','.7').replace('(32 1 1)','(1 1 1)'))
 f=d/'system/controlDict';t=f.read_text();t=re.sub(r'endTime\s+\S+;', f'endTime {end_time};',t);t=re.sub(r'writeInterval\s+\S+;','writeInterval 1;',t);t=re.sub(r'windowSamples\s+\S+;',f'windowSamples {window};',t);f.write_text(t)
 vals={'rho':'1','rhoE':str((1000-8.31446261815324/.028)*1000),'T':'1000','p':str(8.31446261815324/.028*1000),'Tp':'10000','epsilonS':str(mass/(2500*.7)),'dMeanCell':str(diameter),'theta':'1e-12','U':'(0 0 0)','Us':'(0 0 0)','Y_A':'.9','Y_B':'.1','k':'1e-12','omega':'1'}
 for f in (d/'0').iterdir():
  t=f.read_text();t=re.sub(r'internalField.*?;','internalField uniform '+vals[f.name]+';',t,flags=re.S);t=t[:t.index('boundaryField')]+'''boundaryField { left { type zeroGradient; } right { type zeroGradient; } sides { type empty; } }\n''';f.write_text(t)
 f=d/'constant/gasModelProperties';t=f.read_text().replace('(1000 0 -1000000)','(1000 0 0)').replace('(1000 0 2000000)','(1000 0 0)');f.write_text(t)
 f=d/'constant/particleProperties';t=f.read_text();params={'rhoS':2500,'dS':diameter,'dMin':diameter,'dMax':diameter,'parcelMass':mass,'injectionParcelMass':mass,'particleRho':2500,'particleCp':800,'gpuResidentInjectionTheta':1e-12,'gpuResidentInjectionTp':10000}
 for key,value in params.items():t=re.sub(r'\b'+key+r'\s+[^;]+;',f'{key} {value};',t)
 f.write_text(t)
 with (d/'0/gpuResidentStrictParticles.dat').open('wb') as out:
  out.write(b'UGKP_PARTICLES_SCHEMA1_BIN 1 1\n');out.write(struct.pack('<I',1))
  for x in [.35,.5,.5,0,0,0,10000,0,diameter,mass]:out.write(struct.pack('<d',x))
  out.write(struct.pack('<iiQQ',0,1,12345,1))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def field(path):
 return float(re.search(r'internalField\s+uniform\s+([^;]+);',path.read_text()).group(1))
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--binary',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--expect-before',action='store_true')
 args=parser.parse_args();binary=args.binary.resolve();out=args.output.resolve()
 out.mkdir(parents=True,exist_ok=False)
 report={'binary':str(binary),'binary_sha256':sha(binary),'runner_sha256':sha(Path(__file__)),
         'expect_before':args.expect_before,'cases':[]}
 env=dict(os.environ,OMP_NUM_THREADS='4',GAS_UGKP_CUDA_STACK_BYTES='65536')
 for name,mass,diameter,end_time,window in [
   ('failure_final_partial_window',1.,1e-6,1,100),
   ('failure_final_full_window',1.,1e-6,1,1),
   ('failure_before_final_write',1.,1e-6,2,100),
   ('valid_hot_parcel_partial_window',1e-6,.001,1,100),
   ('valid_hot_parcel_full_window',1e-6,.001,1,1)]:
  d=out/name;prepare(d,mass,diameter,end_time,window)
  inputs={str(f.relative_to(d)):sha(f) for f in d.rglob('*') if f.is_file()}
  mesh_command=['blockMesh','-case',str(d)]
  with (d/'mesh.log').open('w') as f:
   mesh=subprocess.run(mesh_command,env=env,stdout=f,stderr=subprocess.STDOUT)
  assert mesh.returncode==0,(name,'blockMesh failed')
  command=[str(binary),'-case',str(d)]
  with (d/'run.log').open('w') as f:
   run=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
  text=(d/'run.log').read_text()
  written=sorted(f.name for f in d.iterdir() if f.is_dir() and f.name.isdigit() and f.name!='0')
  row={'name':name,'command':command,'mesh_command':mesh_command,'exit':run.returncode,
       'input_sha256':inputs,'written_times':written,'log_sha256':sha(d/'run.log')}
  if name.startswith('failure'):
   if args.expect_before:
    passed='1' in written and (d/'1/steadyState.dat').exists()
    if end_time==1:passed=passed and run.returncode==0 and 'samples=1' in text
   else:
    passed=run.returncode!=0 and not written and 'site=30700' in text and 'detailCode=4' in text
    passed=passed and 'Steady sampling window ' not in text and 'Particle steady driver finished' not in text
  else:
   gas=field(d/'1/T');parcel=field(d/'1/Tp');energy=field(d/'1/rhoE')
   # Independent constant-Cp equation for this stationary closed one-cell case.
   # Molecular conductivity is mu*Cp/Pr, with Nu=2 at zero slip.
   cv=1000-8.31446261815324/.028;volume=.7-mass/2500
   alpha=-math.expm1(-12*(1.8e-5*1000/.7)/(2500*800*diameter**2)*2e-5)
   a=mass*800*alpha;target=(volume*cv*1000+a*10000)/(volume*cv+a)
   expected_parcel=10000+alpha*(target-10000)
   heat=mass*800*(parcel-10000);balance=volume*(energy-cv*1000)+heat
   row['numerics']={'gas_temperature':gas,'parcel_temperature':parcel,'expected_gas':target,
                    'expected_parcel':expected_parcel,'parcel_heat_J':heat,'energy_balance_J':balance,'caloric_temperature':energy/cv}
   passed=run.returncode==0 and written==['1'] and (d/'1/steadyState.dat').exists()
   # The ordinary EOS recovery retains its default 1e-10 relative energy
   # tolerance. Check the more sensitive conserved caloric temperature too.
   passed=passed and abs(gas-target)<1e-7 and abs(energy/cv-target)<2e-9 and abs(parcel-expected_parcel)<1e-8 and abs(balance)<2e-9 and heat<0
  row['pass']=passed;report['cases'].append(row)
  (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  print(name,run.returncode,written,'PASS' if passed else 'FAIL',flush=True)
  assert passed,row
 report['pass']=True;(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
