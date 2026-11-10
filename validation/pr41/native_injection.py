"""Real initial-face injection smoke/regression. Run with sourced OpenFOAM.
All inputs are fresh synthetic fixture copies; no production case is modified.
"""
import argparse,csv,importlib.util,json,re,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('native_matrix',ROOT/'validation/molecular_transport/native_matrix.py')
g=module('native_particles',ROOT/'validation/transient_particles/check_native.py')
sys.path.insert(0,str(ROOT/'tool/restart'))
import thermal_particle_restart_current as wire
DT=1e-7
def patch(path,name,body):
 text,n=re.subn(r'\b'+name+r'\s*\{[^}]*\}',name+' { '+body+' }',path.read_text());assert n==1,(path,name);path.write_text(text)
def prepare(case,solver):
 if solver.startswith('gas'):
  g.copy_inputs(ROOT/'validation/shared_boundary/fixtures'/('steady-table' if solver.endswith('Steady') else 'gas-table'),case)
  # Uniform face strips simplify an independent area-uniform population check.
  p=case/'system/blockMeshDict';p.write_text(p.read_text().replace('simpleGrading (1 8.0 1)','simpleGrading (1 1 1)'))
  m.run(['blockMesh','-case',case],case/'mesh.log')
  patch(case/'0/epsilonS','inlet','type fixedValue; value uniform .001;')
  props=case/'constant/particleProperties'
  for k,v in [('parcelMass',1e-13),('particleInjectionVelocity','(1 0 0)'),('dragModel','none'),('particleGasHeatTransferModel','none')]:g.set_entry(props,k,v)
  for k,v in [('gpuResidentPureGasOnly','false'),('gpuResidentParticleCapacity',20000)]:g.set_entry(case/'constant/schedulingProperties',k,v)
  p=case/'constant/fluidProperties';p.write_text(re.sub(r'boundaryLayer\s*\{[^}]*\}','',p.read_text().replace('wallTreatment boundaryLayer','wallTreatment lowRe')))
  if solver.endswith('Steady'):
   for k,v in [('endTime',1),('samplingDeltaT',DT),('maxDeltaT',DT),('writeInterval',1)]:g.set_entry(case/'system/controlDict',k,v)
  else:
   for k,v in [('endTime',DT),('deltaT',DT),('maxDeltaT',DT),('writeInterval',1),('adjustTimeStep','false')]:g.set_entry(case/'system/controlDict',k,v)
  return dict(expected_mass=.001*2800*.002*DT,y=.2,z=.01,x=0)
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())
 fixture='chmt-mixture' if solver=='CHMT' else 'cht-contact'
 m.prepare(ROOT/'validation/molecular_transport/fixtures'/fixture,case,meta[fixture])
 for k,v in [('endTime',DT),('deltaT',DT),('writeInterval',1 if solver=='CHMT' else DT)]:g.set_entry(case/'system/controlDict',k,v)
 if solver=='CHMT':
  p=case/'constant/chmtProperties'
  g.set_entry(p,'couplingInterval',DT);g.set_entry(p,'gasMaxDt',DT)
  p.write_text(p.read_text()+"""
enableParticles true;
particle { volumeMode Coupled; capacity 20000; rhoSolid 2500;
 injectionParcelMass 1e-10; dragModel none; heatTransferModel none; contactModel reboundContact;
 inlet { patch right; temperature 600; theta 0; volumeFraction .001; velocity (-1 0 0); }
}
""")
  p=case/'constant/polyMesh/boundary';p.write_text(re.sub(r'(right\s*\{\s*type\s+)wall',r'\1patch',p.read_text()))
  return dict(expected_mass=.001*2500*DT,y=1,z=1,x=1)
 for k,v in [('gpuResidentPureGasOnly','false'),('gpuResidentParticleCapacity',20000)]:g.set_entry(case/'constant/schedulingProperties',k,v)
 for k,v in [('parcelMass',1e-10),('particleInjectionVelocity','(1 0 0)')]:g.set_entry(case/'constant/particleProperties',k,v)
 patch(case/'0/epsilonS','left','type fixedValue; value uniform .001;')
 return dict(expected_mass=.001*2800*DT,y=1,z=1,x=0)
PROBE=r"""
#include "restart/Checkpoint.C"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 std::ifstream input(std::string(argv[1])+"/state.chmt",std::ios::binary|std::ios::ate);
 chmt::Bytes data(size_t(input.tellg()));input.seekg(0);input.read(reinterpret_cast<char*>(data.data()),data.size());
 chmt::Decoder decoder(data);decoder.offset=72;chmt::ModelConfig model;decoder(model);
 if(decoder.integer(8)!=0x31474643504b4755ULL)return 3;
 ugkp::thermal::visitParticlePhaseConfiguration(decoder,model.particlePhase);
 ugkp::thermal::visitParticleContactConfiguration(decoder,model.particleContact);
 chmt::HostState state;std::string error;if(!chmt::readCheckpoint(argv[1],model,state,error)){std::cerr<<error;return 4;}
 double residual=0;for(double v:state.particleSourceResidualMass)residual+=v;
 std::cout<<std::setprecision(17)<<residual;
}
"""
def inspect(case,solver,expected,out):
 if solver=='CHMT':
  output=case/'chmtOutput';index=max(int(p.stem.split('-')[1]) for p in output.glob('particles-*.csv'))
  with (output/f'particles-{index}.csv').open() as f: rows=list(csv.DictReader(f))
  xyz=np.array([[float(q[k]) for k in ('x','y','z')] for q in rows]);mass=sum(float(q['mass']) for q in rows)
  probe=out/'residual.cpp';probe.write_text(PROBE);exe=out/'residual'
  subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(probe),'-o',str(exe)],check=True)
  residual=float(subprocess.check_output([exe,output/f'checkpoint-{index}'],text=True))
 else:
  last=m.times(case)[-1]
  if solver.startswith('gas'):
   rows=g.parcel(last/'gpuResidentStrictParticles.dat');xyz=np.array([q[:3] for q in rows]);mass=sum(q[9] for q in rows)
  else:
   chunks=[d for _,d in wire.iter_chunks(last/'gpuResidentStrictParticles.dat')]
   xyz=np.concatenate([np.column_stack([d[k] for k in ('px','py','pz')]) for d in chunks]);mass=sum(float(d['pm'].sum()) for d in chunks)
  values=(last/'gpuResidentStrictSourceResidual.dat').read_text().split();residual=sum(map(float,values[3::2]))
 assert len(xyz)>1000,len(xyz)
 assert abs(mass+residual-expected['expected_mass'])<expected['expected_mass']*1e-8,(mass,residual,expected)
 # No lateral injection velocity and tiny duration. Transverse position is the
 # sampled initial face coordinate; x allows the existing owner-cell inset.
 assert np.max(np.abs(xyz[:,0]-expected['x']))<2e-6
 normalized=xyz[:,1:]/np.array([expected['y'],expected['z']])
 assert np.min(normalized)>=-1e-10 and np.max(normalized)<=1+1e-10
 means=normalized.mean(axis=0);variances=normalized.var(axis=0)
 assert np.max(np.abs(means-.5))<.03,means
 assert np.max(np.abs(variances-1/12))<.01,variances
 assert len(np.unique(normalized[:,0]))>len(xyz)*.9
 return dict(status='PASS',count=len(xyz),mass=mass,residual=residual,expected_mass=expected['expected_mass'],mean=means.tolist(),variance=variances.tolist(),x_range=[float(xyz[:,0].min()),float(xyz[:,0].max())],initial_geometry='static fixture; motion retention covered by source review, not this run')
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--prepare-only',action='store_true');a.add_argument('--solvers',nargs='+',choices=['gasUGKP','gasUGKPSteady','CHT','CHMT'],default=['gasUGKP','gasUGKPSteady','CHT','CHMT']);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);report=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {}
 for solver in args.solvers:
  case=out/solver
  try:
   spec=prepare(case,solver) if not case.exists() else json.loads((case/'position-spec.json').read_text())
   (case/'position-spec.json').write_text(json.dumps(spec))
   if args.prepare_only:continue
   m.run([ROOT/'bin'/solver,'-case',case],case/'run.log')
   report[solver]=inspect(case,solver,spec,out)
   report[solver]['binary_sha256']=hashlib.sha256((ROOT/'bin'/solver).read_bytes()).hexdigest()
  except Exception as e:report[solver]={'status':'FAIL','error':repr(e)}
  (out/'results.json').write_text(json.dumps(report,indent=2));print(solver,report.get(solver),flush=True)
 if not args.prepare_only and any(r['status']!='PASS' for r in report.values()):raise SystemExit(1)
if __name__=='__main__':main()
