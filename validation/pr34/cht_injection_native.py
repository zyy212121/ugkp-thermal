"""Real CHT initialization and injection, including the unregistered dictionary."""
import argparse,hashlib,json,sys,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'validation/molecular_transport'),str(ROOT/'tool/restart')]
import native_matrix as matrix
import thermal_particle_restart_current as wire

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
 out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
 report=[]
 # Inlet outward normal is -x. Gas and particle boundary velocities are independent.
 scenarios=[('explicit',0,1),('default',1,None),('zero',1,0),('opposed_in',-1,1),('opposed_out',1,-1)]
 for name,gas,override,scheduled in [(n,g,v,s) for s in (False,True) for n,g,v in scenarios]:
  if scheduled:name='scheduled_'+name
  case=out/name;matrix.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact',case,meta)
  path=case/'constant/schedulingProperties';matrix.entry(path,'gpuResidentPureGasOnly','false');matrix.entry(path,'gpuResidentParticleCapacity',512)
  path=case/'constant/particleProperties';matrix.entry(path,'parcelMass',1e-7)
  if override is not None:path.write_text(path.read_text()+f'\nparticleInjectionVelocity ({override} 0 0);\n')
  path=case/'0/U';path.write_text(path.read_text().replace('(0 0 0)',f'({gas} 0 0)'))
  path=case/'0/epsilonS';path.write_text(path.read_text().replace('left{type zeroGradient;}','left{type fixedValue;value uniform .001;}'))
  if scheduled:
   path=case/'constant/schedulingProperties';matrix.entry(path,'gpuResidentDynamicInlet','true')
   path=case/'constant/particleProperties';path.write_text(path.read_text()+'\ngpuResidentInletPatch left;\n')
   # Deliberately differ from the static .001 boundary so this exercises the schedule.
   (case/'constant/inletPressure.table').write_text('((0 178167.05610328368)(1 178167.05610328368))\n')
   (case/'constant/inletVolumeFraction.table').write_text('((0 .002)(1 .002))\n')
   path=case/'0/p';path.write_text(path.read_text().replace('left{type zeroGradient;}','left{type fixedValue;value uniform 178167.05610328368;}'))
  control=case/'system/controlDict'
  for key,value in [('endTime',2e-6),('deltaT',1e-6),('maxDeltaT',1e-6),('writeInterval',2e-6)]:matrix.entry(control,key,value)
  matrix.run([ROOT/'bin/CHT','-case',case],case/'run.log')
  final=matrix.times(case)[-1];chunks=[d for _,d in wire.iter_chunks(final/'gpuResidentStrictParticles.dat')]
  count=sum(len(d['pm']) for d in chunks);mass=sum(float(d['pm'].sum()) for d in chunks)
  residual=(final/'gpuResidentStrictSourceResidual.dat').read_text().split();assert residual[0]=='UGKP_SOURCE_RESIDUAL_SCHEMA1' and len(residual)==2+2*int(residual[1])
  remaining=sum(float(x) for x in residual[3::2])
  velocity=gas if override is None else override
  expected=(.002 if scheduled else .001)*2800*max(velocity,0)*2e-6
  assert abs(mass+remaining-expected)<1e-15,(name,mass,remaining,expected)
  assert (count>0)==(expected>0),(name,count)
  error=0;momentum=np.zeros(3);kinetic=0.0
  for d in chunks:
   speeds=np.column_stack([d[k] for k in ('pux','puy','puz')])
   momentum+=(d['pm'][:,None]*speeds).sum(axis=0);kinetic+=float((.5*d['pm']*(speeds*speeds).sum(axis=1)).sum())
   error=max(error,float(np.max(np.abs(d['pux']-velocity))))
   assert np.max(np.abs(d['puy']))<1e-6 and np.max(np.abs(d['puz']))<1e-6
  assert error<1e-5,(name,error)
  assert np.max(np.abs(momentum-np.array([mass*velocity,0,0])))<1e-12
  assert abs(kinetic-.5*mass*velocity*velocity)<1e-12
  report.append(dict(case=name,scheduled=scheduled,gas_velocity=gas,override=override,count=count,mass=mass,residual_mass=remaining,expected_mass=expected,maximum_velocity_error=error,momentum=momentum.tolist(),kinetic_energy=kinetic))
 (out/'results.json').write_text(json.dumps({'binary_sha256':hashlib.sha256((ROOT/'bin/CHT').read_bytes()).hexdigest(),'cases':report},indent=2))
 print(json.dumps(report))
if __name__=='__main__':main()
