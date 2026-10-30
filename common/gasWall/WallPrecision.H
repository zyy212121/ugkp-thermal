#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall { namespace detail {
// Typed borrowed views retain their actual storage type. Only scalar values
// are promoted; no mechanism copy, pointer reinterpretation, or allocation.
template<class S,int N> UGKWP_GAS_HD inline WallInput<WallReal,N,S> wallComputationInput(const WallInput<S,N>& src){
    WallInput<WallReal,N,S> dst;
    dst.diagnostics=src.diagnostics;dst.diagnosticEntity=src.diagnosticEntity;dst.diagnosticNode=src.diagnosticNode;
    dst.pressure=src.pressure;dst.temperature=src.temperature;
    dst.matchingDistance=src.matchingDistance;dst.ownerDistance=src.ownerDistance;
    dst.matching.temperature=src.matching.temperature;dst.matching.k=src.matching.k;dst.matching.omega=src.matching.omega;
    for(int d=0;d<3;++d){dst.velocity[d]=src.velocity[d];dst.normal[d]=src.normal[d];dst.matching.velocity[d]=src.matching.velocity[d];}
    for(int j=0;j<N;++j){dst.massFlux[j]=src.massFlux[j];dst.matching.massFraction[j]=src.matching.massFraction[j];dst.model.diffusivity[j]=src.model.diffusivity[j];}
    dst.model.thermo=src.model.thermo;dst.model.mechanism=src.model.mechanism;dst.model.mode=src.model.mode;
    dst.model.molecularViscosity=src.model.molecularViscosity;
    dst.model.viscosity=src.model.viscosity;dst.model.conductivity=src.model.conductivity;dst.model.molecularPrandtl=src.model.molecularPrandtl;
    dst.model.sstCoefficients=src.model.sstCoefficients;dst.quadrature=src.quadrature;dst.coarseReactionIntegral=src.coarseReactionIntegral;
    dst.adiabatic=src.adiabatic;dst.integrateOwnerKSource=src.integrateOwnerKSource;dst.volumeSstClosure=src.volumeSstClosure;
    for(int i=0;i<3;++i)for(int j=0;j<2;++j)dst.sstTangentialGradient[i][j]=src.sstTangentialGradient[i][j];
    return dst;
}
template<class S> UGKWP_GAS_HD inline WallModelConfig<WallReal> wallComputationConfig(const WallModelConfig<S>& src){
    WallModelConfig<WallReal> dst;dst.family=src.family;dst.model=src.model;dst.nodes=src.nodes;dst.maxIterations=src.maxIterations;
    dst.relativeTolerance=src.relativeTolerance;dst.absoluteTolerance=src.absoluteTolerance;dst.stretch=src.stretch;
    dst.equilibriumKappa=src.equilibriumKappa;dst.equilibriumAPlus=src.equilibriumAPlus;
    dst.turbulentPrandtl=src.turbulentPrandtl;dst.turbulentSchmidt=src.turbulentSchmidt;dst.enableSst=src.enableSst;dst.coherentDiscretization=src.coherentDiscretization;return dst;
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
    dst.ownerKBalanceResidual=S(src.ownerKBalanceResidual);dst.ownerKContent=S(src.ownerKContent);dst.ownerKDestruction=S(src.ownerKDestruction);
    dst.kFluxCount=src.kFluxCount;for(int j=0;j<src.kFluxCount;++j){dst.kFluxHeight[j]=S(src.kFluxHeight[j]);dst.kTransportFlux[j]=S(src.kTransportFlux[j]);dst.omegaDiffusiveFlux[j]=S(src.omegaDiffusiveFlux[j]);}
    dst.heatConduction=S(src.heatConduction);dst.heatViscous=S(src.heatViscous);dst.heatTransport=S(src.heatTransport);dst.heatBalanceResidual=S(src.heatBalanceResidual);
    dst.chemistryMismatchAvailable=src.chemistryMismatchAvailable;dst.iterations=src.iterations;dst.residual=S(src.residual);



    // Inspect the values actually published, including narrowing overflow.
    // An unavailable chemistry mismatch is the sole intentional NaN sentinel.
    const S scalars[]={dst.traceTemperature,dst.traceDensity,dst.conductiveHeatFlux,
        dst.ownerOmega,dst.wallKFlux};
    for(int i=0;i<5;++i)if(!gasFinite(scalars[i]))
        return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2810,scalars[i],0,i);
    for(int d=0;d<3;++d){
        if(!gasFinite(dst.traceVelocity[d]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2811,dst.traceVelocity[d],ugkp::diagnostics::Quantity::Velocity,d);
        if(!gasFinite(dst.traction[d]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2814,dst.traction[d],0,d+3);
    }
    for(int j=0;j<N;++j){
        const S values[]={dst.traceMassFraction[j],dst.wallSpeciesFlux[j],dst.matchingSpeciesFlux[j],dst.reactionIntegral[j]};
        for(int i=0;i<4;++i)if(!gasFinite(values[i]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2812,values[i],0,i*N+j);
        if(!gasFinite(dst.speciesBalanceResidual[j]))wallWarning(in,ugkp::diagnostics::Code::ConservationResidual,2813,dst.speciesBalanceResidual[j],std::numeric_limits<S>::max(),0,j);
        if(dst.chemistryMismatchAvailable&&!gasFinite(dst.chemistryMismatch[j]))wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2815,dst.chemistryMismatch[j],std::numeric_limits<S>::max(),0,j);
    }
    if(in.integrateOwnerKSource&&!gasFinite(dst.integratedKSource))wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2816,dst.integratedKSource,std::numeric_limits<S>::max());
    if(!gasFinite(dst.residual))wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2817,dst.residual,std::numeric_limits<S>::max());
    output=dst;return true;
}
}}}
