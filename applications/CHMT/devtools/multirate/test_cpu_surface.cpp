#include "ablation/CpuSurfaceInterface.H"
#include <cassert>
#include <iostream>
using namespace chmt;
int main(){PhysicsConfig p;for(int s=0;s<Ns;++s){p.species[s].R=287;p.species[s].cp0=1000;p.species[s].Tmin=100;p.species[s].Tmax=3000;}
 for(int c=0;c<Nc;++c){p.condensed[c].rho=1000;p.condensed[c].cp0=1000;p.condensed[c].conductivity=2;p.condensed[c].Tmin=100;p.condensed[c].Tmax=3000;}
 p.gasConductivity=2;HostState h;h.solid.resize(1);h.solid[0].condensed[0]=1;h.solid[0].energy=600000;
 h.solidMesh.volumes={.001};h.solidMesh.areaVectors={{1,0,0}};h.gasMesh.areaVectors={{-1,0,0}};h.gasMesh.faceIds={7};h.gasMesh.owner={0};
 GasPrimitive w;w.rho=1;w.temperature=800;w.pressure=1*287*800;w.Y[0]=1;h.gasMesh.volumes={1};h.gas={conservativeGas(w,1,p)};
 h.surface.area={1};h.surface.normal={{1,0,0}};h.surface.gasFace={0};h.surface.solidFace={0};h.surface.solidCell={0};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.filmAux.resize(1);
 CpuSurfaceResult result;std::string error;assert(evaluateCpuSurface(h,p,{w},{},{},{},.01,1,result,error));
 assert(result.wall.size()==1);assert(result.wall[0].temperature>600&&result.wall[0].temperature<800);assert(result.phasePackets.empty());
 assert(result.wall[0].primaryKind==ExchangeKind::GasSolid);assert(h.solid[0].energy==600000);
 std::cout<<"CPU real interface proposal test passed\n";
}
