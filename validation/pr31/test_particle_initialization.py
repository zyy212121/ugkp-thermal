from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]

def test_caloric_import_lifecycle_independence_and_phase_inventory(tmp_path):
    source=tmp_path/'initialization.cpp'
    source.write_text(r'''
#include "thermal/ParticleInitialization.H"
#include "core/HostState.H"
#include "materials/Reaction.H"
#include <cassert>
#include <cmath>
int main(){
 using namespace ugkp::thermal;
 ParticlePhaseConfiguration config;ParticleState<> p;p.mass=.002;p.velocity={1,2,3};
 p.energy=p.mass*ugkwp::ParticleCaloric{}.enthalpy(10000.);
 assert(initializeMobileParticleHistory(p,config));assert(std::abs(p.thermal.temperature-10000.)<1e-10);
 assert(p.thermal.savedVelocity[2]==3&&std::abs(particleSnapshotThermalEnergy(p,1)-p.energy)<1e-12);
 for(int state=1;state<=3;++state){p.state=state;
  for(int k=0;k<8;++k)p.thermal.coldNodeSpecificEnthalpy[k]=float(10000+k);
  for(int k=0;k<64;++k)p.thermal.cold2DNodeSpecificEnthalpy[k]=float(20000+k);
  p.thermal.specificEnthalpyResidual=.123;
  assert(particleSnapshotThermalEnergy(p,3)==p.mass*(10003.5+.123));
  assert(particleSnapshotThermalEnergy(p,4)==p.mass*(20031.5+.123));
  p.thermal.initialized=false;assert(!initializeMobileParticleHistory(p,config));p.thermal.initialized=true;
 }
 chmt::HostState state;state.gasMesh.volumes={.125,.5};
 assert(chmt::gasPhaseVolume(state,0)==.125);state.gasVoidFraction={.8,.9};
 assert(chmt::gasPhaseVolume(state,0)==.1&&chmt::gasPhaseVolume(state,1)==.45);
 auto physics=chmt::PhysicsConfig{};auto& c=physics.condensed[0];
 c.rho=1000;c.cp0=800;c.cp1=.5;c.conductivity=2;c.Tmin=100;c.Tmax=3000;
 chmt::SolidQ solid;solid.condensed[0]=100;solid.energy=100*(800*600+.25*600*600);
 const double expected=std::sqrt(2.*100.*1100./.1);
 assert(std::abs(chmt::materialThermalEffusivity(solid,.1,physics)-expected)<1e-10);
}
''')
    exe=tmp_path/'initialization'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/CHMT'),str(source),'-o',str(exe)],check=True)
    subprocess.run([exe],check=True)
