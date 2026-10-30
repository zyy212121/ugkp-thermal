#pragma once
#include "WallPrecision.H"
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall {
namespace detail {
template<class R> UGKWP_GAS_HD inline R scaledExponentialIntegral(R x){
    const R a=abs(x);return a==0?R(1):-::expm1(-a)/a;
}
// Integral of exp(2*a*x) times the thermal Green function, multiplied by
// expRatio(a)^2. Exponential factors are combined before evaluation.
template<class R> UGKWP_GAS_HD inline bool constantWallHeatingFactor(R a,R b,R& value){
    const R aa=abs(a);
    if(abs(b)>=R(.01)){
        const R da=-::expm1(-aa),db=-::expm1(-abs(b));
        const R ratio=aa==0?R(1):aa/da;
        const R e0=b>0?max(R(0),R(2)*a-b)-R(2)*max(a,R(0)):R(0);
        const R e1=b>0?R(2)*a-b-R(2)*max(a,R(0)):max(b,R(2)*a)-R(2)*max(a,R(0));
        const R e=max(e0,e1);
        const R f0=scaledExponentialIntegral(b>0?R(2)*a-b:R(2)*a);
        const R f1=scaledExponentialIntegral(b>0?R(2)*a:R(2)*a-b);
        const R integral=(::exp(e0-e)*f0-::exp(e1-e)*f1)/db;
        if(!(integral>0))return false;
        value=::exp(e+R(2)*::log(ratio)+::log(integral));
        return gasFinite(value)&&value>=0;
    }
    // Near b=0 the closed-form difference loses digits. Integrate in the
    // exponential boundary-layer coordinate, with an explicit refinement test.
    // The omitted exp(-80) tail is below double precision relative accuracy.
    const R scale=max(R(1),R(2)*aa),end=min(scale,R(80));
    const R ratio=aa==0?R(1):aa/(-::expm1(-aa));
    R previous=-1;
    for(int intervals=16;intervals<=32768;intervals*=2){
        R sum=0;
        for(int i=0;i<=intervals;++i){
            const R z=end*R(i)/R(intervals),x=a>0?R(1)-z/scale:z/scale;
            const R t=R(1)-x;
            const R kernel=b==0?t:(::expm1(b*t)/::expm1(b));
            const R f=::exp(R(2)*a*x-R(2)*max(a,R(0)))*kernel;
            sum+=R(i==0||i==intervals?1:i%2?4:2)*f;
        }
        const R current=sum*end/(R(3)*R(intervals)*scale);
        if(previous>=0&&abs(current-previous)/R(15)<=R(1e-12)*max(abs(current),R(1e-300))){
            value=ratio*(ratio*current);return gasFinite(value)&&value>=0;
        }
        previous=current;
    }
    return false;
}
}
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
    R heating=0;
    if(speed2>0){
        R factor=0;
        if(!detail::constantWallHeatingFactor(a,b,factor))return detail::fail(status,WallCode::NonConvergence);
        heating=mu*speed2/length*factor;
    }
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
