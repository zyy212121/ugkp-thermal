#pragma once
#include "WallPrecision.H"
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall {
// Analytic frozen-composition constant-cp/transport limit, not the reacting SST model.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool evaluateConstantTransportWallImpl(
 const WallInput<R,N,TableReal>& originalInput,const WallModelConfig<R>& c,WallOutput<R,N>& output,WallStatus& status){
    auto in=originalInput;in.model.thermo.diagnostics=in.diagnostics;
    for(int j=0;j<N;++j)if(!detail::wallFraction(in,in.matching.massFraction[j],in.matching.massFraction[j],j))return false;
    if(!detail::validateInput(in,c,status))return false;
    if((in.model.molecularViscosity.model!=MolecularViscosityModel::Constant||in.model.molecularViscosity.prandtlSpecies>=0||in.model.molecularViscosity.diffusion))
        return detail::fail(status,WallCode::Unsupported);
    R m=0,cp=0;for(int j=0;j<N;++j)m+=in.massFlux[j];
    for(int j=0;j<N;++j){bool available=false;const R value=detail::wallProperty<false>(j,in.temperature,in.model.thermo,&available);
        if(!available){detail::fail(status,WallCode::InvalidThermodynamics);return detail::wallFatal(in,ugkp::diagnostics::Code::NonFinite,2412,value);}
        cp+=in.matching.massFraction[j]*value;}
    WallOutput<R,N> out;if(!detail::trace(in,in.matching.massFraction,out,true))return detail::fail(status,WallCode::InvalidThermodynamics);
    const R length=in.matchingDistance,mu=in.model.viscosity,lambda=in.model.molecularPrandtl>0?mu*cp/in.model.molecularPrandtl:in.model.conductivity;
    if(mu==0||lambda==0)return detail::wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2430,mu==0?mu:lambda);
    const R a=m*length/mu,b=m*cp*length/lambda;
    R du[3],normalDifference=0,speed2=0;for(int d=0;d<3;++d){du[d]=in.matching.velocity[d]-in.velocity[d];normalDifference+=du[d]*in.normal[d];}
    for(int d=0;d<3;++d){du[d]-=normalDifference*in.normal[d];speed2+=du[d]*du[d];out.traction[d]=mu/length*detail::expRatio(a)*du[d];}
    // Green-function integral of molecular dissipation. Composite Simpson is
    // used only for the finite-blowing mechanical-heating term; zero-blowing
    // and pure thermal blowing remain exact analytic limits.
    R heating=0;
    if(speed2>0){if(m==0)heating=mu*speed2/(R(2)*length);
      else {const int intervals=64;R sum=0;
        for(int i=0;i<=intervals;++i){const R x=R(i)/intervals;
          const R weight=(i==0||i==intervals)?1:(i%2?4:2);
          const R gradient=detail::expRatio(a)*::exp(a*x)/length;
          const R kernel=detail::abs(b)<R(1e-8)?(R(1)-x): ::exp(-b*x)*(-::expm1(-b*(R(1)-x)))/(-::expm1(-b));
          sum+=weight*mu*speed2*gradient*gradient*kernel;}
        heating=length*sum/(R(3)*intervals);}}
    const R thermalConductance=lambda/length*detail::expRatio(b);
    if(in.adiabatic){
        in.temperature=in.matching.temperature+heating/thermalConductance;
        if(!detail::trace(in,in.matching.massFraction,out,true))return detail::fail(status,WallCode::InvalidThermodynamics);
    }
    out.conductiveHeatFlux=-thermalConductance*(in.matching.temperature-in.temperature)-heating;
    for(int j=0;j<N;++j){out.matchingSpeciesFlux[j]=in.massFlux[j];
        out.chemistryMismatch[j]=in.coarseReactionIntegral?-in.coarseReactionIntegral[j]:std::numeric_limits<R>::quiet_NaN();}
    out.chemistryMismatchAvailable=in.coarseReactionIntegral!=nullptr;
    if(!gasFinite(out.conductiveHeatFlux)||!gasFinite(out.traceDensity)){detail::fail(status,WallCode::InadmissibleState);return detail::wallFatal(in,ugkp::diagnostics::Code::NonFinite,2431,out.conductiveHeatFlux);}
    status.residual=0;output=out;return true;
}
// Keep the original <R,N> entry-point spelling while deducing borrowed tables.
template<class R,int N,class TableReal=R> UGKWP_GAS_HD inline typename std::enable_if<std::is_same<R,WallReal>::value,bool>::type evaluateConstantTransportWall(
 const WallInput<R,N,TableReal>& input,const WallModelConfig<R>& config,
 WallOutput<R,N>& output,WallStatus& status){
 return evaluateConstantTransportWallImpl(input,config,output,status);
}
template<class S,int N> UGKWP_GAS_HD inline typename std::enable_if<!std::is_same<S,WallReal>::value,bool>::type evaluateConstantTransportWall(
 const WallInput<S,N>& input,const WallModelConfig<S>& config,
 WallOutput<S,N>& output,WallStatus& status){
 const auto promoted=detail::wallComputationInput(input);const auto controls=detail::wallComputationConfig(config);
 WallOutput<WallReal,N> result;
 if(!evaluateConstantTransportWallImpl(promoted,controls,result,status))return false;
 if(!detail::publishWallPrecision(input,result,output))return detail::fail(status,WallCode::InadmissibleState);
 return true;
}
}} // namespace
