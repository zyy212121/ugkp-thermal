"""Compile the actual recession-target loop; host coverage, not full OF/CUDA proof."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

APP = (SOURCE_ROOT / 'applications/CHMT')


def test_production_packet_surface_identity_and_material_volume(tmp_path):
    source = (APP / 'materials/CpuMaterialDriver.C').read_text()
    begin = source.index('std::vector<Real> target(state.surface.area.size(),0);')
    end = source.index('for(const auto& phase:surface.phasePackets)', begin)
    loop = source[begin:end]
    assert source.index('packetAddressing.build(base,geometryEndpoint,error)') < source.index('while(elapsed<windowEnd)')
    cpp = r'''
#include "core/HostState.H"
#include "materials/Reaction.H"
#include "materials/CpuMaterialDriver.H"
#include "materials/SurfacePacketAddressing.H"
#include <cassert>
#include <iostream>
using namespace chmt;
namespace chmt {
std::vector<Real> targets(const HostState& state,const PhysicsConfig& p,
 const std::vector<ExchangePacket>& packets,std::string& error,bool terminalMaterial=false){
MaterialFailureOrigin lastFailure=MaterialFailureOrigin::None;
SurfacePacketAddressing packetAddressing;
if(!packetAddressing.build(state,state,error))return {};
PRODUCTION_LOOP
return target;
}
}
int main(){
 PhysicsConfig p;
 for(int c=0;c<Nc;++c){auto& a=p.condensed[c];a.rho=1000+1000*c;a.cp0=1000;a.Tmin=100;a.Tmax=2000;a.conductivity=0;}
 HostState s;s.solid.resize(2);s.solidMesh.volumes={1,1};
 for(auto& q:s.solid){q.condensed[0]=400;q.condensed[1]=800;q.porosity=.99;
  q.energy=q.condensed[0]*condensedE(p.condensed[0],600)+q.condensed[1]*condensedE(p.condensed[1],600);}
 s.surface.area={2,3};s.surface.gasFace={0,1};s.surface.solidCell={0,1};s.surface.persistentId={73,29};s.gasMesh.faceIds={29,73};
 ExchangePacket a;a.kind=ExchangeKind::GasSolid;a.face=73;a.solidCell=0;a.condensed[0]=2;a.condensed[1]=4;
 std::string error;auto t=targets(s,p,{a},error);
 assert(error.empty());assert(t.size()==2);
 auto terminal=targets(s,p,{a},error,true);assert(error.empty()&&terminal==t);
 assert(std::abs(t[0]+.005)<1e-15 && t[1]==0); // occupied=.8, phi=.2; (2/1000+4/2000)/.8
 auto next=a;next.step=1;auto doubled=targets(s,p,{a,next},error);assert(error.empty()&&std::abs(doubled[0]+.01)<1e-15);
 auto missing=a;missing.face=999;error.clear();targets(s,p,{missing},error);assert(!error.empty());
 auto duplicate=s;duplicate.surface.persistentId[1]=73;error.clear();auto duplicateTargets=targets(duplicate,p,{a},error);assert(error.empty()&&std::abs(duplicateTargets[0]+.005)<1e-15);
 error.clear();targets(duplicate,p,{},error);assert(error.empty()); // first insertion retains original resolution
 auto wrong=a;wrong.solidCell=1;error.clear();targets(s,p,{wrong},error);assert(error.empty());
 auto inactive=s;inactive.surface.gasFace[0]=-1;error.clear();targets(inactive,p,{a},error);assert(error.empty());
 auto badDensity=p;badDensity.condensed[1].rho=0;error.clear();targets(s,badDensity,{a},error);assert(!error.empty());
 SurfacePacketAddressing map;error.clear();assert(map.build(s,s,error));
 std::size_t index=99;assert(map.gasSolidIndex(s,a,index,error)&&index==0);
 auto changed=s;changed.surface.gasFace[0]=1;error.clear();assert(map.gasSolidIndex(changed,a,index,error)&&index==0);
 changed=s;changed.surface.persistentId[0]=99;error.clear();assert(map.gasSolidIndex(changed,a,index,error)&&index==0);
 changed=s;changed.surface.solidCell[0]=1;error.clear();assert(map.gasSolidIndex(changed,a,index,error)&&index==0);
 changed=s;changed.gasMesh.topologyHash=1;error.clear();assert(map.gasSolidIndex(changed,a,index,error)&&index==0);
 for(int change=0;change<4;++change){changed=s;
  if(change==0)changed.surface.persistentId[1]=99;
  if(change==1)changed.surface.gasFace[1]=7;
  if(change==2)changed.surface.solidCell[1]=0;
  if(change==3)changed.solidMesh.topologyHash=1;
  error.clear();assert(map.build(s,changed,error));}
 // A fresh candidate may use new, validated ordering and moving metrics.
 changed=s;changed.surface.persistentId={29,73};changed.surface.solidCell={1,0};changed.surface.area={4,5};
 error.clear();assert(map.build(changed,changed,error));assert(map.gasSolidIndex(changed,a,index,error)&&index==1);
 std::cout<<"surface ID, duplicate/missing, donor and material-volume contracts passed\n";
}
'''.replace('PRODUCTION_LOOP', loop)
    src=tmp_path/'recession.cpp';src.write_text(cpp);binary=tmp_path/'recession'
    build=subprocess.run(['g++','-std=c++17','-O2','-I',str(APP), '-I', str(VALIDATION_ROOT / 'solvers/CHMT'),'-I',str(APP.parents[1]/'common'),str(src),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
