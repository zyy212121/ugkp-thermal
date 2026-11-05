"""CHMT checkpoint stores the complete common particle history, including f32 profiles."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/"applications/CHMT"
WRITERS=Path(__file__).resolve().parent/"fixtures/checkpoint_writers"
PRE=r'''
#include "restart/Checkpoint.H"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iterator>
using namespace chmt;
void fill(HostState& s) {
 s.time=.125;s.gasTemperature={850};s.gasMesh.owner={0};
 auto b=std::make_shared<ugkp::boundary::Definition>();b->faces.resize(1);
 b->faces[0].function[ugkp::boundary::Temperature]=0;
 b->functions={{ugkp::boundary::FunctionKind::Constant,ugkp::boundary::Bounds::Clamp,0,1,0}};
 b->terms={{0,300}};s.gasMesh.gasBoundary=b;
 s.particles.resize(4);
 for(int j=0;j<4;++j){auto& p=s.particles[j];
  p.id=91+j;p.rng=12345+j;p.cell=0;p.face=j?0:-1;p.state=j;
  p.position={.1*j,.2*j,.3*j};p.velocity={j+1.,j+2.,j+3.};p.contactRelativeVelocity={-1.,2.,-3.};
  p.mass=.001*(j+1);p.energy=100+j;p.diameter=.0001*(j+1);p.contactAge=.03*j;
  p.contactMechanicalEnergy=17+j;p.contactDuration=.1;p.contactArea=.000002;
  auto& h=p.thermal;h.initialized=true;h.status=1;h.temperature=1000+j;h.theta=.25+j;
  h.savedVelocity[0]=-3-j;h.savedVelocity[1]=2+j;h.savedVelocity[2]=.25*j;
  h.specificEnthalpyResidual=(j%2?-1:1)*1e-9;h.contactMaximumArea=.00001*(j+1);h.contactPeakFraction=.4;
  h.coldFrozenArea=.000003;h.coldContactAge=.04*j;h.cold2DFrozenArea=.000005;
  for(int k=0;k<8;++k){h.coldNodeSpecificEnthalpy[k]=std::nextafter(float(1e6+j*1000+k),float(INFINITY));
   h.coldRingSolidMass[k]=1e-10*(k+1);h.cold2DRingContactAge[k]=.001*k;}
  for(int k=0;k<64;++k)h.cold2DNodeSpecificEnthalpy[k]=std::nextafter(float(2e6+j*1000+k),float(INFINITY));
 }
}
std::string bytes(const std::string& path){std::ifstream f(path+"/state.chmt",std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
'''


def build(tmp,body,checkpoint=None,name="probe"):
    source=tmp/(name+".cpp");source.write_text(PRE+body);exe=tmp/name
    command=["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror",
        "-I"+str(APP),"-I"+str(APP/"restart"),"-I"+str(ROOT/"common"),str(source),
        str(checkpoint or APP/"restart/Checkpoint.C"),"-o",str(exe)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    return exe


def test_populated_history_roundtrip_with_boundary_and_gas_temperature(tmp_path):
    exe=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;model.physics.gasTransportFingerprint=1234;
 HostState initial,restored;fill(initial);std::string error;
 assert(writeCheckpoint(argv[1],model,initial,error));const auto raw=bytes(argv[1]);
 assert(readCheckpoint(argv[1],model,restored,error));assert(restored.gasTemperature==initial.gasTemperature);
 assert(restored.gasMesh.gasBoundary&&restored.particles.size()==4);
 for(int j=0;j<4;++j){const auto& a=initial.particles[j];const auto& b=restored.particles[j];
  assert(a.id==b.id&&a.rng==b.rng&&a.state==b.state&&a.face==b.face&&a.cell==b.cell);
  assert(a.position.x==b.position.x&&a.position.y==b.position.y&&a.position.z==b.position.z);
  assert(a.velocity.x==b.velocity.x&&a.contactRelativeVelocity.z==b.contactRelativeVelocity.z);
  assert(a.mass==b.mass&&a.energy==b.energy&&a.contactMechanicalEnergy==b.contactMechanicalEnergy);
  assert(a.contactAge==b.contactAge&&a.contactDuration==b.contactDuration&&a.contactArea==b.contactArea);
  const auto& x=a.thermal;const auto& y=b.thermal;
  assert(y.initialized&&x.status==y.status&&x.temperature==y.temperature&&x.theta==y.theta);
  for(int k=0;k<3;++k)assert(x.savedVelocity[k]==y.savedVelocity[k]);
  assert(x.specificEnthalpyResidual==y.specificEnthalpyResidual&&x.contactMaximumArea==y.contactMaximumArea&&x.contactPeakFraction==y.contactPeakFraction);
  assert(x.coldFrozenArea==y.coldFrozenArea&&x.coldContactAge==y.coldContactAge&&x.cold2DFrozenArea==y.cold2DFrozenArea);
  for(int k=0;k<8;++k){assert(x.coldNodeSpecificEnthalpy[k]==y.coldNodeSpecificEnthalpy[k]);
   assert(x.coldRingSolidMass[k]==y.coldRingSolidMass[k]&&x.cold2DRingContactAge[k]==y.cold2DRingContactAge[k]);}
  for(int k=0;k<64;++k)assert(x.cold2DNodeSpecificEnthalpy[k]==y.cold2DNodeSpecificEnthalpy[k]);
 }
 assert(writeCheckpoint(argv[1],model,restored,error));assert(bytes(argv[1])==raw);
}
''')
    subprocess.run([exe,tmp_path/"checkpoint"],check=True)


def test_schema9_with_variable_transport_keeps_legacy_state(tmp_path):
    path=WRITERS/"schema9.C"
    writer=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;model.physics.gasTransportFingerprint=1234;
 HostState state;fill(state);std::string error;assert(writeCheckpoint(argv[1],model,state,error));}
''',path,"old-writer")
    subprocess.run([writer,tmp_path/"old"],check=True)
    reader=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;model.physics.gasTransportFingerprint=1234;
 HostState state;std::string error;assert(readCheckpoint(argv[1],model,state,error));
 assert(state.gasTemperature.size()==1&&state.gasTemperature[0]==850&&state.particles.size()==4);
 for(int j=0;j<4;++j){assert(state.particles[j].id==91u+j&&state.particles[j].energy==100+j);
  assert(!state.particles[j].thermal.initialized&&state.particles[j].thermal.specificEnthalpyResidual==0);}
 model.physics.gasTransportFingerprint=4321;HostState other;
 assert(!readCheckpoint(argv[1],model,other,error));assert(error.find("transport identity")!=std::string::npos);
}
''',name="new-reader")
    subprocess.run([reader,tmp_path/"old"],check=True)


def test_particle_model_identity_is_checked_even_before_first_injection(tmp_path):
    exe=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;model.physics.enableParticles=true;
 model.particlePhase.capacity=100;model.particlePhase.rhoSolid=2700;
 model.particleContact.model=4;model.particleContact.latentHeat=9e5;
 HostState empty,restored;std::string error;
 assert(writeCheckpoint(argv[1],model,empty,error));
 assert(readCheckpoint(argv[1],model,restored,error));
 auto tuned=model;tuned.particlePhase.capacity=1000;tuned.particlePhase.csrCellLocalPath=0;
 tuned.particlePhase.particleBlockThreads=256;tuned.particlePhase.heavyReductionMode=1;
 assert(readCheckpoint(argv[1],tuned,restored,error));
 auto changed=model;changed.particleContact.model=3;
 assert(!readCheckpoint(argv[1],changed,restored,error));assert(error.find("particle model identity")!=std::string::npos);
 changed=model;changed.particleContact.latentHeat*=2;
 assert(!readCheckpoint(argv[1],changed,restored,error));
 changed=model;changed.particlePhase.rhoSolid*=2;
 assert(!readCheckpoint(argv[1],changed,restored,error));
 changed=model;changed.physics.enableParticles=false;
 assert(!readCheckpoint(argv[1],changed,restored,error));
}
''',name="model-identity")
    subprocess.run([exe,tmp_path/"checkpoint"],check=True)


def test_schema10_complete_history_is_preserved(tmp_path):
    path=WRITERS/"schema10.C"
    writer=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;HostState state;fill(state);
 std::string error;assert(writeCheckpoint(argv[1],model,state,error));}
''',path,"schema10-writer")
    subprocess.run([writer,tmp_path/"old10"],check=True)
    reader=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig model;HostState state,reference;fill(reference);
 std::string error;assert(readCheckpoint(argv[1],model,state,error));
 assert(state.particles.size()==reference.particles.size());
 for(int j=0;j<4;++j){const auto& a=state.particles[j].thermal;const auto& b=reference.particles[j].thermal;
 assert(a.initialized&&a.temperature==b.temperature&&a.coldContactAge==b.coldContactAge);
 for(int k=0;k<64;++k)assert(a.cold2DNodeSpecificEnthalpy[k]==b.cold2DNodeSpecificEnthalpy[k]);}
}
''',name="schema10-reader")
    subprocess.run([reader,tmp_path/"old10"],check=True)
