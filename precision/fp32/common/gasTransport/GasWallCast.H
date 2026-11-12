#pragma once
#include "../gasWall/WallModelTypes.H"
namespace ugkwp {
template<class D,class S> UGKWP_GAS_HD inline gaswall::WallModelConfig<D> castWallModelConfig(const gaswall::WallModelConfig<S>& s){ gaswall::WallModelConfig<D> d; d.numerics=gaswall::castWallIterationControls<D>(s.numerics);
d.family=s.family;
d.model=s.model;
d.nodes=s.nodes;
d.maxIterations=s.maxIterations;
d.relativeTolerance=s.relativeTolerance;
d.absoluteTolerance=s.absoluteTolerance;
// Startup scheme has already resolved defaults; preserve every explicit value.
d.stretch=s.stretch;
d.turbulentPrandtl=s.turbulentPrandtl;
d.turbulentSchmidt=s.turbulentSchmidt;
d.equilibriumKappa=s.equilibriumKappa;
d.equilibriumAPlus=s.equilibriumAPlus;
d.localStressCorrection=s.localStressCorrection;
d.enableSst=s.enableSst;
d.coherentDiscretization=s.coherentDiscretization;
return d;}
template<class D,class S,int N> inline void copyWallExchange(const gaswall::WallExchangeOutput<S,N>& s,gaswall::WallExchangeOutput<D,N>& d){
d.traceTemperature=s.traceTemperature;
d.traceDensity=s.traceDensity;
d.conductiveHeatFlux=s.conductiveHeatFlux;
d.integratedKSource=s.integratedKSource;
d.ownerOmega=s.ownerOmega;
d.wallKFlux=s.wallKFlux;
d.ownerKContent=s.ownerKContent;
d.ownerKDestruction=s.ownerKDestruction;
d.ownerKBalanceResidual=s.ownerKBalanceResidual;
d.kFluxCount=s.kFluxCount;
d.heatConduction=s.heatConduction;
d.heatViscous=s.heatViscous;
d.heatTransport=s.heatTransport;
d.heatBalanceResidual=s.heatBalanceResidual;
d.chemistryMismatchAvailable=s.chemistryMismatchAvailable;
d.iterations=s.iterations;
d.residual=s.residual;
for(int i=0;i<3;++i){d.traceVelocity[i]=s.traceVelocity[i];d.traction[i]=s.traction[i];}
for(int i=0;i<N;++i){d.traceMassFraction[i]=s.traceMassFraction[i];d.wallSpeciesFlux[i]=s.wallSpeciesFlux[i];d.matchingSpeciesFlux[i]=s.matchingSpeciesFlux[i];d.reactionIntegral[i]=s.reactionIntegral[i];d.chemistryMismatch[i]=s.chemistryMismatch[i];d.speciesBalanceResidual[i]=s.speciesBalanceResidual[i];}}
}
