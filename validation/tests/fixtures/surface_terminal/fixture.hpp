#include "ablation/CpuSurfaceInterface.H"
#include <cassert>
#include <iostream>
using namespace chmt;
PhysicsConfig physics(){PhysicsConfig p;
 for(auto& s:p.species){s.R=287;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
 for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.conductivity=2;c.Tmin=100;c.Tmax=3000;}
 p.liquid=p.condensed[0];p.gasConductivity=2;p.gasViscosity=.01;p.liquidViscosity=1;return p;}
HostState host(const PhysicsConfig& p){HostState h;h.solid.resize(1);h.solid[0].condensed[0]=1;h.solid[0].energy=599310;
 h.solidMesh.volumes={.001};h.solidMesh.areaVectors={{1,0,0}};h.gasMesh.areaVectors={{-1,0,0}};h.gasMesh.faceIds={7};h.gasMesh.owner={0};h.gasMesh.volumes={1};
 h.surface.area={1};h.surface.normal={{1,0,0}};h.surface.gasFace={0};h.surface.solidFace={0};h.surface.solidCell={0};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.filmAux.resize(1);
 GasPrimitive w;w.rho=1;w.temperature=813.37;w.pressure=287*w.temperature;w.Y[0]=1;h.gas={conservativeGas(w,1,p)};return h;}
GasPrimitive bulk(const HostState& h,const PhysicsConfig& p){GasPrimitive w;assert(recoverGas(h.gas[0],1,p,w));return w;}
SurfacePhysicsInput input(const HostState& h,const PhysicsConfig& p){SurfacePhysicsInput in;in.hasSolid=true;in.bulk=bulk(h,p);in.solid=in.solidBase=h.solid[0];in.solidVolume=.001;in.area=in.gasArea=in.solidArea=1;in.gasDistance=in.solidDistance=.5;in.normal=in.gasNormal=in.solidNormal={1,0,0};in.dt=.01;in.gasInventory=h.gas[0];return in;}
void vectorOut(Vec3 v){std::cout<<v.x<<' '<<v.y<<' '<<v.z<<' ';}
void resultOut(const SurfacePhysicsResult& r){const auto& g=r.gasState;
 std::cout<<g.rho<<' '<<g.temperature<<' '<<g.pressure<<' '<<g.soundSpeed<<' ';vectorOut(g.velocity);for(auto v:g.Y)std::cout<<v<<' ';
 for(auto v:r.gasSpecies)std::cout<<v<<' ';for(auto v:r.poreSpecies)std::cout<<v<<' ';for(auto v:r.poreDarcy)std::cout<<v<<' ';for(auto v:r.poreSweep)std::cout<<v<<' ';for(auto v:r.condensed)std::cout<<v<<' ';
 for(Real v:{r.phaseMass,r.gasEnergy,r.poreEnergy,r.phaseEnergy,r.gasAdvective,r.gasConductive,r.gasPressureWork,r.gasViscousWork,r.phaseAdvective,r.phaseConductive,r.phasePressureWork,r.phaseViscousWork,r.topKineticAdvection,r.bottomKineticAdvection,r.topTemperature,r.bottomTemperature,r.solidSpeed,r.topSpeed,r.radiationFlux,r.thermalResidual,r.gasWallSpeed,r.solidWallSpeed,r.liquidTopNormal,r.liquidBottomNormal,r.solidRecessionResidual,r.filmVolumeResidual})std::cout<<v<<' ';
 vectorOut(r.shear);vectorOut(r.gasShear);vectorOut(r.traction);std::cout<<r.wet<<'\n';}
