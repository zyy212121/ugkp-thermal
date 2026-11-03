#pragma once
#include "ParticleReservoirReconstruction.H"
namespace mixtureParticles {
template<class State> struct MacroReservoir {
 __device__ bool member(const State&s,int i,int c)const{return mixtureParticles::member(s,i,c);}
 __device__ double volume(const State&s,int c)const{return s.epsGPrev[c]*s.V[c];}
 __device__ double mass(const State&s,int c)const{return volume(s,c)*s.rho[c];}
 __device__ double density(const State&s,int c)const{return s.rho[c];}
 __device__ double velocity(const State&s,int c,int d)const{return d==0?s.Ux[c]:d==1?s.Uy[c]:s.Uz[c];}
 __device__ double momentum(const State&s,int c,int d)const{return d==0?s.rhoUx[c]:d==1?s.rhoUy[c]:s.rhoUz[c];}
 __device__ double energy(const State&s,int c)const{return s.rhoE[c];}
 __device__ double temperature(const State&s,int c)const{return s.Tgas[c];}
 __device__ double species(const State&s,int c,int j)const{return s.gasSpecies.rho[j*s.nCells+c];}
 template<class Drag> __device__ double beta(const State&s,int c,int i,double eps)const{return mixtureParticles::beta<Drag>(s,c,i,eps);}
 __device__ double heatFraction(const State&s,int c,int i,double dt)const{return mixtureParticles::heatFraction(s,c,i,dt);}
 __device__ void apply(State&s,int c,const double* momentum,double energy,bool=false)const{
  const double v=volume(s,c);s.rhoUx[c]-=momentum[0]/v;s.rhoUy[c]-=momentum[1]/v;s.rhoUz[c]-=momentum[2]/v;s.rhoE[c]-=energy/v;recoverGasPrimitiveCell(s,c);
 }
};
// A selector is geometric only. The same transaction below is used for cells
// and wall reservoirs; no wall-private drag/heat update exists.
template<class State,int Ns,class Selector> struct NodeReservoir {
 ugkwp::ParticleThermalReservoir<Ns,double>& gas;const Selector& selector;
 const double* frozenBetaDt=nullptr;
 double thermalExchange=0,reverseMomentum[3]{},reverseEnergy=0;
 __device__ bool member(const State&s,int i,int c)const{return mixtureParticles::member(s,i,c)&&selector(s,i);}
 __device__ double volume(const State&,int)const{return gas.volume;}
 __device__ double mass(const State&,int)const{return gas.mass;}
 __device__ double density(const State&,int)const{return gas.mass/gas.volume;}
 __device__ double velocity(const State&,int,int d)const{return gas.momentum[d]/gas.mass;}
 __device__ double momentum(const State&,int,int d)const{return gas.momentum[d]/gas.volume;}
 __device__ double energy(const State&,int)const{return gas.energy/gas.volume;}
 __device__ double temperature(const State&,int)const{return gas.temperature;}
 __device__ double species(const State&,int,int j)const{return gas.speciesMass[j]/gas.volume;}
 __device__ void fractions(double* y)const{for(int j=0;j<Ns;++j)y[j]=gas.speciesMass[j]/gas.mass;}
 __device__ double viscosity(const State&s)const{double y[Ns];fractions(y);return ugkwp::molecularViscosity<Ns>(s.gasViscosity,s.gasMu,gas.temperature,y);}
 template<class Drag> __device__ double beta(const State&s,int c,int i,double eps)const{
  double slip2=0;const double p[3]={s.pux[i],s.puy[i],s.puz[i]};for(int d=0;d<3;++d){const double v=velocity(s,c,d)-p[d];slip2+=v*v;}
  const ugkwpGpuDrag::DragInput input={density(s,c),viscosity(s),eps,s.rhoSolid,s.pd[i],::sqrt(slip2),1e-300,s.dragParameter0,s.dragParameter1,s.dragParameter2,s.dragParameter3};
  return fmax(0.,Drag::inverseResponseTime(input));
 }
 template<class Drag> __device__ double betaTime(const State&s,int c,int i,double eps,double dt)const{
  if(frozenBetaDt&&frozenBetaDt[i]>=0)return frozenBetaDt[i];
  return beta<Drag>(s,c,i,eps)*dt;
 }
 __device__ double heatFraction(const State&s,int c,int i,double dt)const{
  double y[Ns];fractions(y);const double mu=viscosity(s);const double cp=ugkwp::mixtureSpecificCp(y,gas.temperature,s.gasSpecies.thermo);
  const double pr=ugkwp::molecularPrandtl<Ns>(s.gasViscosity,s.gasPrClamped,y);
  double slip2=0;const double p[3]={s.pux[i],s.puy[i],s.puz[i]};for(int d=0;d<3;++d){const double v=velocity(s,c,d)-p[d];slip2+=v*v;}
  const double re=density(s,c)*s.pd[i]*::sqrt(slip2)/fmax(mu,1e-30);
  const double nu=ugkwp::ranzMarshallNuFromPr(fmax(0.,re),pr);
  double lambda=mu*cp/pr;
  if constexpr(ugkwp::GasDirectConductivity<State>::value)if(ugkwp::gasHasDirectConductivity(s))lambda=s.gasThermalConductivity;
  const double rate=6*nu*lambda/(s.particleThermalCapacity*s.pd[i]*s.pd[i]+1e-300);
  return -expm1(-fmax(0.,rate)*dt);
 }
 __device__ void apply(State&s,int c,const double* momentum,double energy,bool thermal=false){
   if(thermal)thermalExchange-=energy;
   for(int d=0;d<3;++d)reverseMomentum[d]-=momentum[d];reverseEnergy-=energy;
  for(int d=0;d<3;++d)gas.momentum[d]-=momentum[d];gas.energy-=energy;
  double kinetic=0;for(int d=0;d<3;++d)kinetic+=.5*gas.momentum[d]*gas.momentum[d]/gas.mass;
  const auto status=ugkwp::invertMixtureEnergy(gas.speciesMass,gas.energy-kinetic,s.gasSpecies.thermo,s.gasSpecies.thermoControls,gas.temperature);
  if(status!=ugkwp::ThermoStatus::Success){s.gasSpecies.cellStatus[c]=static_cast<int>(status);
   using namespace ugkp::diagnostics;recordFatal(s.diagnostics,makeRecord(Code::InvalidPhysicalState,30700,0,0,EntityKind::Cell,c,-1,int(status)));}
 }
};
}
