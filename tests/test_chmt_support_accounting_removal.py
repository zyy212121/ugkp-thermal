"""Actual reduced-model physical outputs retain FP64 values after force removal."""
from pathlib import Path
import io
import subprocess
import tarfile
ROOT = Path(__file__).resolve().parents[1]
BASE = '0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668'
SOURCE = r'''
#include "LIFECYCLE_HEADER"
#include "coupling/Coordinator.H"
#include <cstdio>
void scalar(Real x){std::printf("%a ",x);}
void vector(Vec3 v){scalar(v.x);scalar(v.y);scalar(v.z);}
int main(int argc,char**){
 auto p=lifecyclePhysics();p.material.enableMelting=true;
 for(Real temperature:{500.,600.}){
  auto h=lifecycleState(p,.02);auto g=primitive(p,temperature);
  p.gravity={};p.gasViscosity=0;g.velocity={};
  h.surface.baseVelocity.assign(2,Vec3{});
  CpuSurfaceResult surface;std::string error;
  require(evaluateCpuSurface(h,p,std::vector<GasPrimitive>(2,g),{},{},{},.001,1,surface,error),error);
  for(const auto& q:surface.phasePackets){
   if(argc>1)require(mag(q.momentum)==0,"unused phase momentum still computed");
   scalar(q.mass);scalar(q.energy);scalar(q.advective);scalar(q.conductive);scalar(q.pressureWork);scalar(q.viscousWork);scalar(q.liquidKineticAdvection);
   for(auto v:q.species)scalar(v);for(auto v:q.condensed)scalar(v);
  }
  for(const auto& r:surface.physics){scalar(r.phaseMass);scalar(r.phaseEnergy);scalar(r.topTemperature);scalar(r.solidSpeed);scalar(r.topSpeed);vector(r.shear);vector(r.gasState.velocity);scalar(r.gasState.pressure);}
  p.gravity={};p.gasViscosity=0;h.surface.baseVelocity.assign(2,Vec3{});g.velocity={};
  const auto step=advanceLifecycle(h,p,std::vector<GasPrimitive>(2,g),.001);
  scalar(step.gasMass);scalar(step.gasEnergy);scalar(step.phaseMass);
  for(const auto& q:step.state.solid){scalar(q.energy);scalar(q.porosity);for(auto v:q.condensed)scalar(v);for(auto v:q.pore)scalar(v);}
  for(const auto& q:step.state.film){scalar(q.mass);scalar(q.enthalpy);for(auto v:q.species)scalar(v);}
  for(const auto& a:step.state.filmAux){scalar(a.pressure);scalar(a.thickness);scalar(a.temperature);vector(a.meanVelocity);vector(a.topVelocity);scalar(a.kineticEnergy);scalar(a.solidFront);scalar(a.gasFront);}
  for(auto x:step.state.solidMesh.points)vector(x);for(auto x:step.state.gasMesh.points)vector(x);
 }
 // Both packet producers preserve phase energy and mass for signed phase rates.
 for(Real rate:{-.01,0.,.01}){
  SurfacePhysicsInput in;in.hasSolid=true;in.dt=.001;in.area=2;in.solidArea=3;in.gasArea=4;
  in.normal=in.solidNormal=in.gasNormal={0,0,1};in.aux.thickness=.02;in.baseVelocity={.2,.3,0};
  in.filmBase.mass=1;in.filmBase.species[0]=1;in.filmBase.enthalpy=1000;
  SurfacePhysicsResult r;r.wet=true;r.phaseMass=rate;r.phaseAdvective=7;r.phaseConductive=9;
  r.phasePressureWork=11;r.phaseViscousWork=13;r.bottomKineticAdvection=17;
  r.shear={.3,.4,0};r.gasState.pressure=100000;p.gravity={1,2,3};
  SurfacePacketIdentity id;id.gasCell=id.solidCell=id.filmFace=0;ExchangePacket out[3];
  require(assembleSurfacePackets(in,r,id,p,out),"actual signed phase assembly");
  if(argc>1)require(mag(out[2].momentum)==0,"unused assembled phase momentum still computed");
  scalar(out[2].mass);scalar(out[2].energy);scalar(out[2].advective);scalar(out[2].conductive);
  scalar(out[2].pressureWork);scalar(out[2].viscousWork);scalar(out[2].liquidKineticAdvection);
 }
#ifdef CURRENT_API
 if(argc>1){
  ExchangePacket q;q.kind=ExchangeKind::GasSolid;q.gasCell=q.solidCell=0;q.momentum={3,4,5};
  require(accountAssembledPacket(q,Tolerances{}),"actual assembly markers");
  require(q.consumerMask==(ConsumeGas|ConsumeSolid),"actual consumer markers changed");
 }
#endif
}
'''

def test_removed_support_red_green_and_physical_parity(tmp_path):
    baseline = tmp_path / 'base'; baseline.mkdir()
    archive = subprocess.check_output(['git', 'archive', BASE, 'applications/CHMT', 'common'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar: tar.extractall(baseline, filter='data')
    outputs = []
    for label, root in [('base', baseline), ('current', ROOT)]:
        app = root / 'applications/CHMT'
        fixture = tmp_path / (label + '-lifecycle.H')
        fixture.write_text((app/'devtools/multirate/test_cpu_phase_lifecycle.cpp').read_text().replace('int main(int argc,char** argv)', 'int lifecycle_test_main(int argc,char** argv)'))
        source = tmp_path / (label + '.cpp'); source.write_text(SOURCE.replace('LIFECYCLE_HEADER', str(fixture)))
        exe = tmp_path / (label+'-probe')
        command = ['g++', '-std=c++17', '-O1', '-ffp-contract=off', '-I'+str(app), '-I'+str(app/'devtools/multirate'), '-I'+str(root/'common'), str(source)]
        command += [str(app / name) for name in ['ablation/CpuSurfaceInterface.C', 'film/CpuFilmDriver.C', 'materials/MaterialTransport.C', 'mesh/Geometry.C']]
        if label == 'current': command += ['-DCURRENT_API']
        command += ['-o', str(exe)]
        build = subprocess.run(command, capture_output=True, text=True)
        (tmp_path/(label+'.build.log')).write_text(' '.join(command)+'\n'+build.stdout+build.stderr)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        (tmp_path/(label+'.parity.log')).write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stderr
        outputs.append(run.stdout)
        removal = subprocess.run([str(exe), 'removed'], capture_output=True, text=True)
        (tmp_path/(label+'.removal.log')).write_text(removal.stdout+removal.stderr)
        if label == 'base':
            assert removal.returncode != 0
            assert 'unused phase momentum still computed' in removal.stderr
        else: assert removal.returncode == 0, removal.stderr
    assert outputs[0] == outputs[1], 'physical mass/energy/profile/geometry FP64 outputs changed'


def test_legacy_sst_nonfinite_slots_do_not_change_actual_packet_state(tmp_path):
    source=tmp_path/'legacy.cpp'
    source.write_text(r'''
#include "coupling/Coordinator.H"
#include <cassert>
#include <cmath>
#include <limits>
using namespace chmt;
int main(){HostState clean,legacy;legacy.budget.turbulenceOmegaConstraint=NAN;legacy.budget.turbulenceInventory=INFINITY;legacy.budget.turbulenceProduction=-INFINITY;legacy.budget.supportImpulse={NAN,INFINITY,-INFINITY};legacy.budget.boundaryEnergy=NAN;
 clean.gas.resize(1);clean.solid.resize(1);clean.gas[0].mass=clean.gas[0].species[0]=1;clean.gas[0].energy=1000;clean.solid[0].condensed[0]=1;clean.solid[0].energy=1000;
 legacy.gas=clean.gas;legacy.solid=clean.solid;
 ExchangePacket a;a.kind=ExchangeKind::GasSolid;a.gasCell=a.solidCell=0;a.mass=.25;a.species[0]=a.condensed[0]=.25;a.energy=a.advective=13;a.momentum={3,4,5};auto b=a;
 std::string error;
 assert(consumePacketTransaction(a,&clean.gas[0],&clean.solid[0],nullptr,nullptr,error));
 assert(consumePacketTransaction(b,&legacy.gas[0],&legacy.solid[0],nullptr,nullptr,error));
 assert(clean.gas[0].mass==1.25&&clean.solid[0].condensed[0]==.75);
 assert(clean.gas[0].energy==1013&&clean.solid[0].energy==987&&clean.gas[0].momentum.x==3);
 assert(clean.gas[0].mass==legacy.gas[0].mass&&clean.gas[0].energy==legacy.gas[0].energy&&clean.gas[0].momentum.x==legacy.gas[0].momentum.x&&clean.solid[0].energy==legacy.solid[0].energy);
 assert(a.consumerMask==b.consumerMask&&a.consumerMask==(ConsumeGas|ConsumeSolid));
 assert(std::isnan(legacy.budget.turbulenceOmegaConstraint)&&legacy.budget.turbulenceInventory==INFINITY&&std::isnan(legacy.budget.boundaryEnergy));
}
''')
    app=ROOT/'applications/CHMT';exe=tmp_path/'legacy'
    subprocess.run(['g++','-std=c++17','-O1','-ffp-contract=off','-I'+str(app),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
