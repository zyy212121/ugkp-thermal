"""Disappearing-film composition residuals warn without changing exact transfer."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_disappearing_film_composition_warning(tmp_path):
    source=tmp_path/'phase.cpp'
    source.write_text(r'''#include "ablation/InterfaceMath.H"
#include <cassert>
#include <iostream>
using namespace chmt;
int main(){
 ugkp::diagnostics::DeviceContext diagnostics{};PhysicsConfig p;p.diagnostics=&diagnostics;p.tolerances.diagnostics=&diagnostics;
 p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1;
 SurfacePhysicsInput in;in.hasSolid=true;in.hasFilm=true;in.dt=1;in.area=1;in.gasArea=1;
 in.filmBase.mass=1;in.filmBase.species[0]=.75;in.filmBase.species[1]=.25;in.filmBase.enthalpy=37;
 SurfacePhysicsResult result;result.wet=true;result.phaseMass=-2;
 SurfacePacketIdentity id;id.gasCell=id.solidCell=id.filmFace=0;id.oldFilmPV=2;
 ExchangePacket packets[3];DeviceStatus status;
 assert(assembleSurfacePackets(in,result,id,p,packets,&status));
 assert(packets[2].mass==-1&&packets[2].condensed[0]==-1);
 assert(packets[2].species[0]==-.75&&packets[2].species[1]==-.25&&packets[2].energy==-35);
 assert(diagnostics.warnings[0].count==2&&diagnostics.warnings[0].sample.sourceSiteId==30901);
 assert(status.code==0&&!diagnostics.fatalClaim);
 assert(diagnostics.warnings[0].sample.observed==-.25);
 diagnostics={};status={};in.filmBase.species[0]=1;in.filmBase.species[1]=0;
 assert(assembleSurfacePackets(in,result,id,p,packets,&status)&&diagnostics.warnings[0].count==0&&status.code==0);
 diagnostics={};status={};in.filmBase.species[0]=std::numeric_limits<Real>::infinity();packets[2].mass=123;
 assert(!assembleSurfacePackets(in,result,id,p,packets,&status));
 assert(status.code==static_cast<int>(ErrorCode::Nonfinite)&&packets[2].mass==123&&diagnostics.warnings[0].count==0);
 std::cout<<"PASS finite phase-composition warning, exact transfer preserved, invalid arithmetic fails\n";
}
''')
    exe=tmp_path/'phase'
    subprocess.run(['g++','-std=c++17','-O1','-g','-DUGKWP_GAS_SPECIES=2',
                    '-fsanitize=undefined','-fno-sanitize-recover=undefined',
                    '-I'+str(SOURCE_ROOT/'applications/CHMT'),'-I'+str(SOURCE_ROOT/'common'),
                    str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
