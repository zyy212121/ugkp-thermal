#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall {
// Analytic frozen-composition constant-cp/transport limit, not the reacting SST model.
template<class R,int N> UGKWP_GAS_HD inline bool evaluateConstantTransportWall(
 const WallInput<R,N>& in,const WallModelConfig<R>& c,WallOutput<R,N>& output,WallStatus& status){
    if(!detail::validateInput(in,c,status))return false;
    if(c.enableSst||in.model.mode==GasMode::MixtureChemistry)return detail::fail(status,WallCode::Unsupported);
    R m=0,cp=0;for(int j=0;j<N;++j)m+=in.massFlux[j];
    for(int j=0;j<N;++j){const auto& sp=in.model.thermo.species[j];
        if(sp.model!=SpeciesThermoModel::LinearCp||in.model.thermo.coefficients[sp.coefficientOffset+1]!=0)return detail::fail(status,WallCode::Unsupported);
        if(detail::abs(in.massFlux[j]-m*in.matching.massFraction[j])>c.absoluteTolerance+c.relativeTolerance*detail::abs(m))return detail::fail(status,WallCode::Unsupported);
        cp+=in.matching.massFraction[j]*speciesCp(j,in.temperature,in.model.thermo);}
    WallOutput<R,N> out;if(!detail::trace(in,in.matching.massFraction,out))return detail::fail(status,WallCode::InvalidThermodynamics);
    const R length=in.matchingDistance,mu=in.model.viscosity,lambda=in.model.molecularPrandtl>0?mu*cp/in.model.molecularPrandtl:in.model.conductivity;
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
    out.conductiveHeatFlux=-lambda/length*detail::expRatio(b)*(in.matching.temperature-in.temperature)-heating;
    for(int j=0;j<N;++j){out.matchingSpeciesFlux[j]=in.massFlux[j];
        out.chemistryMismatch[j]=in.coarseReactionIntegral?-in.coarseReactionIntegral[j]:std::numeric_limits<R>::quiet_NaN();}
    out.chemistryMismatchAvailable=in.coarseReactionIntegral!=nullptr;
    if(!gasFinite(out.conductiveHeatFlux))return detail::fail(status,WallCode::InadmissibleState);
    status.residual=0;output=out;return true;
}
}} // namespace
