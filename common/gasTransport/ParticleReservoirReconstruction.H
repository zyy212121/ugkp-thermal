#pragma once
#include "ParticleReservoirRelaxation.H"
namespace ugkwp {
template<int Ns,class R> struct ParticleReservoirShape {
 R volumeFraction=0,density=0,temperature=0,velocity[3]{},massFraction[Ns]{};
};
template<int Ns,class R> struct ParticleThermalReservoir:ParticleGasReservoir<R> {
 R volume=0,speciesMass[Ns]{},temperature=0;
};
// Reconstruct extensive gas inventories before any local exchange. Species,
// momentum and total energy match the macro inventory; a realizability limiter
// limits only subcell fluctuations. It never clips a physical parcel exchange.
template<int Ns,class R,class Table>
UGKWP_GAS_HD inline bool reconstructParticleReservoirs(
 const ParticleGasReservoir<R>& macro,const R* speciesMass,R gasVolume,R initialTemperature,
 const ParticleReservoirShape<Ns,R>* shape,int nodes,const Table& thermo,
 const ThermoInversionControls<R>& controls,ParticleThermalReservoir<Ns,R>* out){
 if(nodes<=0||!(gasVolume>0&&macro.mass>0))return false;
 R volumeSum=0;
 for(int i=0;i<nodes;++i){if(!(shape[i].volumeFraction>=0))return false;volumeSum+=shape[i].volumeFraction;out[i]={};out[i].volume=gasVolume*shape[i].volumeFraction;}
 if(::fabs(volumeSum-R(1))>R(64)*std::numeric_limits<R>::epsilon())return false;
 R density[Ns],lo=0,hi=std::numeric_limits<R>::max();
 for(int j=0;j<Ns;++j){
  if(!(speciesMass[j]>=0))return false;
  density[j]=speciesMass[j]/gasVolume;
  lo=lo>R(thermo.species[j].minTemperature)?lo:R(thermo.species[j].minTemperature);
  hi=hi<R(thermo.species[j].maxTemperature)?hi:R(thermo.species[j].maxTemperature);
  R raw=0;for(int i=0;i<nodes;++i)raw+=out[i].volume*shape[i].density*shape[i].massFraction[j];
  for(int i=0;i<nodes;++i){out[i].speciesMass[j]=raw>0?speciesMass[j]*(out[i].volume*shape[i].density*shape[i].massFraction[j])/raw:speciesMass[j]*shape[i].volumeFraction;out[i].mass+=out[i].speciesMass[j];}
 }
 if(!(hi>=lo))return false;
 R mean[3]{},shift[3]{},massSum=0;
 for(int i=0;i<nodes;++i){if(!(out[i].mass>=0)||(out[i].volume>0&&out[i].mass==0))return false;massSum+=out[i].mass;for(int d=0;d<3;++d)mean[d]+=out[i].mass*shape[i].velocity[d];}
 for(int d=0;d<3;++d){mean[d]/=massSum;shift[d]=macro.momentum[d]/massSum;}
 R fluctuationEnergy=0,meanKinetic=0;
 for(int d=0;d<3;++d)meanKinetic+=R(.5)*massSum*shift[d]*shift[d];
 for(int i=0;i<nodes;++i)for(int d=0;d<3;++d){const R v=shape[i].velocity[d]-mean[d];fluctuationEnergy+=R(.5)*out[i].mass*v*v;}
 const R minimumEnergy=gasVolume*mixtureEnergy(density,lo,thermo);
 const R thermalBudget=macro.energy-meanKinetic-minimumEnergy;
 if(!(thermalBudget>=0))return false;
 R velocityScale=1;
 if(fluctuationEnergy>thermalBudget&&fluctuationEnergy>0)velocityScale=::sqrt(thermalBudget/fluctuationEnergy);
 R totalKinetic=0;
 for(int i=0;i<nodes;++i){R kinetic=0;for(int d=0;d<3;++d){const R v=shift[d]+velocityScale*(shape[i].velocity[d]-mean[d]);out[i].momentum[d]=out[i].mass*v;kinetic+=R(.5)*out[i].mass*v*v;}out[i].energy=kinetic;totalKinetic+=kinetic;}
 R baseTemperature=initialTemperature;
 if(invertMixtureEnergy(density,(macro.energy-totalKinetic)/gasVolume,thermo,controls,baseTemperature)!=ThermoStatus::Success)return false;
 // Thermodynamic inversion tolerances must not become inventory errors.
 const R baseCorrection=macro.energy-totalKinetic-gasVolume*mixtureEnergy(density,baseTemperature,thermo);
 R rawDifference=-baseCorrection;
 for(int i=0;i<nodes;++i)rawDifference+=mixtureEnergy(out[i].speciesMass,shape[i].temperature,thermo)-mixtureEnergy(out[i].speciesMass,baseTemperature,thermo);
 R thermalScale=1;
 for(int i=0;i<nodes;++i){
  if(out[i].mass==0)continue;
  const R base=mixtureEnergy(out[i].speciesMass,baseTemperature,thermo)+out[i].mass*baseCorrection/massSum;
  const R delta=mixtureEnergy(out[i].speciesMass,shape[i].temperature,thermo)-base-out[i].mass*rawDifference/massSum;
  const R floor=mixtureEnergy(out[i].speciesMass,lo,thermo),ceiling=mixtureEnergy(out[i].speciesMass,hi,thermo);
  if(delta<0)thermalScale=thermalScale<(base-floor)/(-delta)?thermalScale:(base-floor)/(-delta);
  if(delta>0)thermalScale=thermalScale<(ceiling-base)/delta?thermalScale:(ceiling-base)/delta;
 }
 for(int i=0;i<nodes;++i){
  if(out[i].mass==0)continue;
  const R base=mixtureEnergy(out[i].speciesMass,baseTemperature,thermo)+out[i].mass*baseCorrection/massSum;
  const R delta=mixtureEnergy(out[i].speciesMass,shape[i].temperature,thermo)-base-out[i].mass*rawDifference/massSum;
  const R internal=base+thermalScale*delta;
  out[i].energy+=internal;out[i].temperature=baseTemperature;
  if(invertMixtureEnergy(out[i].speciesMass,internal,thermo,controls,out[i].temperature)!=ThermoStatus::Success)return false;
 }
 return true;
}
}
