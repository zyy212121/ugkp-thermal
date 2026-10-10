#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall { namespace detail {
// Typed borrowed views retain their actual storage type. Only scalar values
// are promoted; no mechanism copy, pointer reinterpretation, or allocation.
template<class S,int N> UGKWP_GAS_HD inline WallInput<WallReal,N,S> wallComputationInput(const WallInput<S,N>& src){
    WallInput<WallReal,N,S> dst;
    dst.pressure=src.pressure;dst.temperature=src.temperature;
    dst.matchingDistance=src.matchingDistance;dst.ownerDistance=src.ownerDistance;
    dst.matching.temperature=src.matching.temperature;dst.matching.k=src.matching.k;dst.matching.omega=src.matching.omega;
    for(int d=0;d<3;++d){dst.velocity[d]=src.velocity[d];dst.normal[d]=src.normal[d];dst.matching.velocity[d]=src.matching.velocity[d];}
    for(int j=0;j<N;++j){dst.massFlux[j]=src.massFlux[j];dst.matching.massFraction[j]=src.matching.massFraction[j];dst.model.diffusivity[j]=src.model.diffusivity[j];}
    dst.model.thermo=src.model.thermo;dst.model.mechanism=src.model.mechanism;dst.model.mode=src.model.mode;
    dst.model.viscosity=src.model.viscosity;dst.model.conductivity=src.model.conductivity;dst.model.molecularPrandtl=src.model.molecularPrandtl;
    dst.model.sstCoefficients=src.model.sstCoefficients;dst.quadrature=src.quadrature;dst.coarseReactionIntegral=src.coarseReactionIntegral;
    return dst;
}
template<class S> UGKWP_GAS_HD inline WallModelConfig<WallReal> wallComputationConfig(const WallModelConfig<S>& src){
    WallModelConfig<WallReal> dst;dst.family=src.family;dst.model=src.model;dst.nodes=src.nodes;dst.maxIterations=src.maxIterations;
    dst.relativeTolerance=src.relativeTolerance;dst.absoluteTolerance=src.absoluteTolerance;dst.stretch=src.stretch;
    dst.turbulentPrandtl=src.turbulentPrandtl;dst.turbulentSchmidt=src.turbulentSchmidt;dst.enableSst=src.enableSst;return dst;
}
template<class S,int N> UGKWP_GAS_HD inline bool publishWallPrecision(const WallInput<S,N>& in,const WallOutput<WallReal,N>& src,WallOutput<S,N>& output){
    WallOutput<S,N> dst;
    dst.traceTemperature=S(src.traceTemperature);dst.traceDensity=S(src.traceDensity);dst.conductiveHeatFlux=S(src.conductiveHeatFlux);
    for(int d=0;d<3;++d){dst.traceVelocity[d]=S(src.traceVelocity[d]);dst.traction[d]=S(src.traction[d]);}
    for(int j=0;j<N;++j){dst.traceMassFraction[j]=S(src.traceMassFraction[j]);dst.wallSpeciesFlux[j]=S(src.wallSpeciesFlux[j]);
        dst.matchingSpeciesFlux[j]=S(src.matchingSpeciesFlux[j]);dst.reactionIntegral[j]=S(src.reactionIntegral[j]);
        dst.speciesBalanceResidual[j]=S(WallReal(dst.matchingSpeciesFlux[j])-WallReal(dst.wallSpeciesFlux[j])-WallReal(dst.reactionIntegral[j]));
        dst.chemistryMismatch[j]=src.chemistryMismatchAvailable?S(WallReal(dst.reactionIntegral[j])-WallReal(in.coarseReactionIntegral[j])):std::numeric_limits<S>::quiet_NaN();}
    dst.integratedKSource=S(src.integratedKSource);dst.ownerOmega=S(src.ownerOmega);dst.wallKFlux=S(src.wallKFlux);
    dst.chemistryMismatchAvailable=src.chemistryMismatchAvailable;dst.iterations=src.iterations;dst.residual=S(src.residual);
    if(!gasFinite(dst.traceTemperature)||!gasFinite(dst.traceDensity)||!gasFinite(dst.conductiveHeatFlux)
        ||!gasFinite(dst.integratedKSource)||!gasFinite(dst.ownerOmega)||!gasFinite(dst.wallKFlux)||!gasFinite(dst.residual))return false;
    for(int d=0;d<3;++d)if(!gasFinite(dst.traceVelocity[d])||!gasFinite(dst.traction[d]))return false;
    for(int j=0;j<N;++j)if(!gasFinite(dst.traceMassFraction[j])||!gasFinite(dst.wallSpeciesFlux[j])
        ||!gasFinite(dst.matchingSpeciesFlux[j])||!gasFinite(dst.reactionIntegral[j])||!gasFinite(dst.speciesBalanceResidual[j])
        ||(dst.chemistryMismatchAvailable&&!gasFinite(dst.chemistryMismatch[j])))return false;
    output=dst;return true;
}
}}}
