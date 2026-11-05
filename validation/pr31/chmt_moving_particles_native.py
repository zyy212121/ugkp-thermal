"""Actual CHMT reacting-SST recession + attached particle histories + restart."""
import argparse,json,re,shutil,subprocess
from pathlib import Path
from chmt_particles_native import m,ROOT,table
PROBE=r'''#include "restart/Checkpoint.C"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
 if(argc!=2)return 2;
 std::ifstream input(std::string(argv[1])+"/state.chmt",std::ios::binary|std::ios::ate);
 chmt::Bytes data(size_t(input.tellg()));input.seekg(0);input.read(reinterpret_cast<char*>(data.data()),data.size());
 chmt::Decoder decoder(data);decoder.offset=72;chmt::ModelConfig model;decoder(model);
 if(decoder.integer(8)!=0x31474643504b4755ULL)return 3;
 ugkp::thermal::visitParticlePhaseConfiguration(decoder,model.particlePhase);
 ugkp::thermal::visitParticleContactConfiguration(decoder,model.particleContact);
 chmt::HostState state;std::string error;
 if(!chmt::readCheckpoint(argv[1],model,state,error)){std::cerr<<error;return 4;}
 std::cout<<std::setprecision(17)<<"{\"time\":"<<state.time<<",\"particles\":[";
 for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(i)std::cout<<',';
  const auto center=state.gasMesh.faceCentres.at(p.face);
  const auto area=state.gasMesh.areaVectors.at(p.face);const double distance=chmt::dot(p.position-center,area)/chmt::mag(area);
  std::cout<<"{\"id\":"<<p.id<<",\"distance\":"<<distance<<",\"wall_y\":"<<center.y
    <<",\"particle_y\":"<<p.position.y<<",\"age\":"<<p.contactAge<<",\"state\":"<<p.state<<",\"status\":"<<p.thermal.status
    <<",\"owner_matches\":"<<(p.cell==state.gasMesh.owner[p.face]?"true":"false")<<"}";
 }
 state.ledger.clear();chmt::Encoder physical;physical(state);physical(state.gasTemperature);
 for(const auto& p:state.particles)ugkp::thermal::visitParticleThermalHistory(physical,p.thermal);
 std::cout<<"],\"physical_checksum\":"<<chmt::checksum(physical.data.data(),physical.data.size())<<"}\n";
}
'''
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 probe=out/'geometry.cpp';probe.write_text(PROBE);exe=out/'geometry'
 subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(probe),'-o',str(exe)],check=True)
 solver=ROOT/'applications/CHMT/.build/native/CHMT';meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-moving']
 reports=[]
 for model in ('reboundContact','coldWall1D','coldWall2D'):
  case=out/model;m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-moving',case,meta)
  m.entry(case/'system/controlDict','endTime',2e-6)
  boundary=(case/'constant/polyMesh/boundary').read_text()
  wall=re.search(r'\bwall\s*\{([^}]+)',boundary)[1];face=int(re.search(r'startFace\s+(\d+)',wall)[1])
  # Existing cold law at 2500 K; retain a nonuniform internal profile.
  cp=lambda t:(153.5+.0031*(t-2327.))/.101961276
  h=1273.*2317.+(.5*(1273.+cp(2327.))+1.16e6/20.)*20.+.5*(cp(2337.)+cp(2500.))*163.
  profiles=''
  if model=='coldWall1D':profiles='coldNodeSpecificEnthalpy ('+' '.join(str(h+j) for j in range(8))+'); coldRingSolidMass (0 0 0 0 0 0 0 0);'
  if model=='coldWall2D':profiles='cold2DNodeSpecificEnthalpy ('+' '.join(str(h+j) for j in range(64))+'); cold2DRingContactAge (0.02 0.01 0 0 0 0 0 0);'
  properties=case/'constant/chmtProperties'
  properties.write_text(properties.read_text()+f'''\nenableParticles true;
particle {{volumeMode Coupled; capacity 8; rhoSolid 2500; dragModel none; heatTransferModel RanzMarshall; contactModel {model};
contactCoeffs {{interfaceResistance 0.00001;}}}}
particles {{attached {{id "42"; rng "1234567"; cell 0; face {face}; state 2;
position (0.25 0 0.005); velocity (0 0 0); mass 1e-8; diameter .001; temperature 2500;
contactAge .0001; contactDuration .001; contactArea 0;
thermalHistory {{temperature 2500; savedVelocity (0 1 0); contactMaximumArea 1e-6; contactPeakFraction .4;
specificEnthalpyResidual .125; {profiles}}}
}}}}\n''')
  seconds=m.run([solver,'-case',case],case/'run.log')
  output=case/'chmtOutput';last=max(int(p.name.split('-')[1]) for p in output.glob('checkpoint-*'))
  initial=json.loads(subprocess.check_output([exe,output/'checkpoint-0'],text=True));final=json.loads(subprocess.check_output([exe,output/f'checkpoint-{last}'],text=True))
  q=final['particles'][0];assert q['status']==1 and q['state']!=0 and q['owner_matches']
  assert abs(q['distance'])<1e-14 and abs(q['wall_y']-initial['particles'][0]['wall_y'])>1e-13,q
  assert abs(q['age']-(initial['particles'][0]['age']+final['time']))<1e-16,q
  restart=out/(model+'-restart');shutil.copytree(case,restart,ignore=shutil.ignore_patterns('chmtOutput'))
  shutil.copytree(output/'checkpoint-1',restart/'initial-checkpoint')
  d=restart/'constant/chmtProperties';d.write_text(d.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
  m.run([solver,'-case',restart],restart/'restart.log')
  identical=(output/f'checkpoint-{last}/state.chmt').read_bytes()==(restart/'chmtOutput'/f'checkpoint-{last}/state.chmt').read_bytes()
  resumed=json.loads(subprocess.check_output([exe,restart/'chmtOutput'/f'checkpoint-{last}'],text=True))
  assert final==resumed,(final,resumed)
  # The wall heat counters restart at zero. Differencing their cumulative J
  # can change a ledger value by one ulp; all accepted physical histories must
  # still be bit-identical, including every cold-profile node.
  ledger_delta=0.
  for left,right in zip(table(output/f'exchange-{last}.csv'),table(restart/'chmtOutput'/f'exchange-{last}.csv')):
   for key in left:
    try:ledger_delta=max(ledger_delta,abs(float(left[key])-float(right[key])))
    except ValueError:assert left[key]==right[key]
  assert ledger_delta<1e-20,ledger_delta
  reports.append(dict(model=model,seconds=seconds,initial=initial,final=final,restart_checkpoint_byte_identical=identical,maximum_ledger_difference=ledger_delta))
 (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
