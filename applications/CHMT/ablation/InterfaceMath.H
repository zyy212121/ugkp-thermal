#ifndef CHMT_ABLATION_INTERFACEMATH_H
#define CHMT_ABLATION_INTERFACEMATH_H
#include "film/FilmMath.H"
#include "gpu/CommonGasTraction.H"
#include "../../../common/gasWall/WallModel.H"
namespace chmt {
CHMT_HD inline bool interfaceFailure(const PhysicsConfig& p,unsigned site,Real value,int component=-1,
    ugkp::diagnostics::Code code=ugkp::diagnostics::Code::InvalidPhysicalState,Real reference=0) {
    using namespace ugkp::diagnostics;
    if(p.diagnostics)recordFatal(p.diagnostics,makeRecord(finite(value)?code:Code::NonFinite,site,38,0,
        EntityKind::Face,-1,component,value,reference,0,Quantity::Accuracy));
    return false;
}

CHMT_HD inline Real interfacePartialPressure(Real pressure, const Real* Y,
    const PhysicsConfig& p, int species) {
    Real R=0;
    for (int s=0; s<Ns; ++s) R+=Y[s]*p.species[s].R;
    return R>0 ? pressure*Y[species]*p.species[species].R/R : undefinedValue();
}
#include "InterfaceCompositionArithmetic.H"
// One producer per diagnostic site keeps component/value provenance while the
// template policy suppresses only private mathematical trial diagnostics.
template<bool Report>
CHMT_HD inline bool compositionInputFailure(const PhysicsConfig& p,Real value,int component=-1) {
    if(Report)return interfaceFailure(p,3840,value,component);
    return false;
}
template<bool Report>
CHMT_HD inline bool compositionArithmeticFailure(const PhysicsConfig& p,Real value,int component=-1) {
    if(Report)return interfaceFailure(p,3841,value,component,ugkp::diagnostics::Code::NonFinite);
    return false;
}
CHMT_HD inline bool compositionConfigurationFailure(const PhysicsConfig& p,Real value,int component=-1,
    ugkp::diagnostics::Code code=ugkp::diagnostics::Code::InvalidPhysicalState) {
    return interfaceFailure(p,3842,value,component,code);
}
template<bool Report>
CHMT_HD inline bool compositionResidualPolicy(const Real* Y,const Real* bulk,const Real* Js,
    Real pressure,Real T,Real distance,const PhysicsConfig& p,Real* residual,Real* jacobian) {
    using namespace composition_detail;
    Real gasR[Ns]{},R=0,J=0,S=0,a[Ns]{},difference[Ns]{};
    for(int i=0;i<Ns;++i){
        gasR[i]=p.species[i].R;
        if(!finite(Y[i]))return compositionInputFailure<Report>(p,Y[i],i);
        if(!finite(bulk[i]))return compositionInputFailure<Report>(p,bulk[i],i);
        if(!finite(Js[i]))return compositionInputFailure<Report>(p,Js[i],i);
        if(!finite(gasR[i])||gasR[i]<=0)return compositionInputFailure<Report>(p,gasR[i],i);
        if(!finite(p.gasDiffusivity[i])||p.gasDiffusivity[i]<0)
            return compositionInputFailure<Report>(p,p.gasDiffusivity[i],i);
    }
    if(!scaledSum(gasR,Y,R)||!scaledSum(Js,nullptr,J))
        return compositionArithmeticFailure<Report>(p,HUGE_VAL,-1);
    const Real denominators[]={R,pressure,T,distance};
    for(int component=0;component<4;++component)
        if(!finite(denominators[component])||!(denominators[component]>0))
            return compositionInputFailure<Report>(p,denominators[component],component);
    for(int i=0;i<Ns;++i){
        // Evaluate rho*D/distance without an overflowing intermediate product.
        if(!diffusionScale(pressure,p.gasDiffusivity[i],T,distance,R,a[i])
            ||!sumFinite(Y[i],-bulk[i],difference[i]))
            return compositionArithmeticFailure<Report>(p,HUGE_VAL,i);
        Real term=0;
        if(!productFinite(a[i],difference[i],term)||!sumFinite(S,term,S))
            return compositionArithmeticFailure<Report>(p,HUGE_VAL,i);
    }
    for(int i=0;i<Ns;++i){
        Real first=0,second=0,diffusion=0,advective=0,value=0;
        if(!productFinite(a[i],difference[i],first)||!productFinite(Y[i],S,second)
            ||!sumFinite(first,-second,diffusion)||!productFinite(J,Y[i],advective)
            ||!sumFinite(advective,diffusion,value)||!sumFinite(value,-Js[i],residual[i]))
            return compositionArithmeticFailure<Report>(p,HUGE_VAL,i);
        if(jacobian)for(int j=0;j<Ns;++j){
            Real diagonal=0,off=0,ratio=0,thermo=0;
            if(i==j && (!sumFinite(J,a[i],diagonal)||!sumFinite(diagonal,-S,diagonal)))
                return compositionArithmeticFailure<Report>(p,HUGE_VAL,i);
            bool thermoFinite=productFinite(diffusion,gasR[j],thermo)&&quotientFinite(thermo,R,thermo);
            if(!thermoFinite)thermoFinite=quotientFinite(gasR[j],R,ratio)&&productFinite(diffusion,ratio,thermo);
            if(!productFinite(Y[i],a[j],off)||!thermoFinite||!sumFinite(diagonal,-off,value)
                ||!sumFinite(value,-thermo,jacobian[i*Ns+j]))
                return compositionArithmeticFailure<Report>(p,HUGE_VAL,i);
        }
    }
    return true;
}
CHMT_HD inline bool compositionResidual(const Real* Y,const Real* bulk,const Real* Js,
    Real pressure,Real T,Real distance,const PhysicsConfig& p,Real* residual,Real* jacobian) {
    return compositionResidualPolicy<true>(Y,bulk,Js,pressure,T,distance,p,residual,jacobian);
}
#include "InterfaceComposition.H"
CHMT_HD inline bool interfaceComposition(const Real* Js,const Real* suppliedBulk,
    Real pressure,Real T,Real distance,const PhysicsConfig& p,
    Real absoluteFluxTolerance,Real* out) {
 Real bulk[Ns]{},J=0,scale=0,sum=0,maxD=0;
 const Real denominators[]={pressure,T,distance};
 for(int c=0;c<3;++c)if(!finite(denominators[c])||!(denominators[c]>0))
  return compositionInputFailure<true>(p,denominators[c],c+1);
 for(int i=0;i<Ns;++i){
  bulk[i]=suppliedBulk[i];if(!clipPhysicsMassFraction(p,bulk[i],3805,i))return false;
  if(!finite(Js[i]))return compositionConfigurationFailure(p,Js[i],i);
  if(!finite(p.gasDiffusivity[i])||p.gasDiffusivity[i]<0)
   return compositionConfigurationFailure(p,p.gasDiffusivity[i],i);
  if(!finite(p.species[i].R)||!(p.species[i].R>0))return compositionConfigurationFailure(p,p.species[i].R,i);
  scale=maxValue(scale,absValue(Js[i]));sum+=bulk[i];maxD=maxValue(maxD,p.gasDiffusivity[i]);
 }
 if(!composition_detail::scaledSum(Js,nullptr,J))
  return compositionConfigurationFailure(p,J,-1,ugkp::diagnostics::Code::NonFinite);
 if(!finite(absoluteFluxTolerance)||absoluteFluxTolerance<0)
  return compositionConfigurationFailure(p,absoluteFluxTolerance);
 if(!finite(p.tolerances.relativeMass)||p.tolerances.relativeMass<0)
  return compositionConfigurationFailure(p,p.tolerances.relativeMass);
 if(!(sum>0))return compositionConfigurationFailure(p,sum);
 if(!closeEnough(sum,1,1e-13,p.tolerances.relativeMass))
  reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3806,sum-1,0);
 Real relativeScale=0;for(int i=0;i<Ns;++i)if(scale>0)relativeScale+=absValue(Js[i]/scale);
 Real tolerance=0,relative=0;
 if(!composition_detail::productFinite(p.tolerances.relativeMass,relativeScale,relative)
     ||!composition_detail::productFinite(relative,scale,relative)
     ||!composition_detail::sumFinite(absoluteFluxTolerance,relative,tolerance))tolerance=HUGE_VAL;
 if(maxD==0)return composition_detail::linearComposition(Js,bulk,J,pressure,T,distance,p,tolerance,out);
 return composition_detail::scalarSolve(Js,bulk,pressure,T,distance,p,tolerance,out);
}
// Both the primitive builder and the temperature preflight consume this same
// selected species model; retain one source producer for its diagnostic site.
CHMT_HD inline bool interfaceSpeciesModelFailure(const PhysicsConfig& p,Real value,
    int component,Real reference=0) {
    return interfaceFailure(p,3843,value,component,
        ugkp::diagnostics::Code::MissingOperator,reference);
}
CHMT_HD inline bool makeInterfacePrimitive(const Real* suppliedY, Real T, Real pressure,
    Vec3 tangentialVelocity, Vec3 normal, Real normalSpeed, Real J,
    const PhysicsConfig& p, GasPrimitive& out,bool* numericalFailure=nullptr) {
    (void)numericalFailure;
    Real Y[Ns]{};
    for(int species=0;species<Ns;++species){Y[species]=suppliedY[species];
        if(!clipPhysicsMassFraction(p,Y[species],3809,species))return false;}
    Real R=0, cv=0, sum=0;
    const GasThermoAdapter<Ns> thermo(p.species);
    const auto thermoView=thermo.view(p.diagnostics);
    for (int s=0; s<Ns; ++s) {
        if(!finite(p.species[s].R)||!(p.species[s].R>0)||!validSpecies(p.species[s]))return interfaceSpeciesModelFailure(p,
            (!finite(p.species[s].R)||p.species[s].R<=0)?p.species[s].R:static_cast<Real>(p.species[s].model),s,p.species[s].R);
        if(!finite(Y[s])||Y[s]<0)return interfaceFailure(p,3856,Y[s],s);
        if(!finite(T)||T<=0)return interfaceFailure(p,3857,T,-1);
        R+=Y[s]*p.species[s].R;
        cv+=Y[s]*ugkwp::speciesCv(s,T,thermoView);
        sum+=Y[s];
    }
    if(!finite(cv)||cv<=0){using namespace ugkp::diagnostics;
        if(p.diagnostics)recordFatal(p.diagnostics,makeRecord(finite(cv)?Code::InvalidPhysicalState:Code::NonFinite,
            3861,38,0,EntityKind::Face,-1,-1,cv,T,0,Quantity::HeatCapacity));return false;}
    if(!finite(sum)||!finite(pressure)||!finite(R)||pressure<=0||R<=0)return false;
    if(!closeEnough(sum,1,1e-13,p.tolerances.relativeMass))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3817,sum-1,0);
    GasPrimitive state;
    state.temperature=T; state.pressure=pressure; state.rho=pressure/(R*T);
    state.velocity=tangentialVelocity-normal*dot(tangentialVelocity,normal)
        +normal*(normalSpeed+J/state.rho);
    state.soundSpeed=::sqrt((1+R/cv)*R*T);
    for (int s=0; s<Ns; ++s) state.Y[s]=Y[s];
    if ((!finite(state.velocity) || !finite(state.rho) || state.rho<=0)) return false;
    out=state; return true;
}
// Borrowed stage data. The matching state and geometric quadrature are rebuilt
// against the current stage; canonical mechanism storage remains with the caller.
struct GasWallClosureContext {
    ugkwp::GasMechanismView<Real,Ns> mechanism{};
    ugkwp::gaswall::WallMatchingState<Real,Ns> matching{};
    ugkwp::gaswall::WallOwnerQuadrature<Real> quadrature{};
    // Matching pressure closes EOS/BVP/film thermodynamics. Mechanical pressure
    // is the local gas-face pressure used by momentum and its paired sweep work.
    Real matchingPressure=0,mechanicalPressure=0,matchingDistance=0,ownerDistance=0;
    void* workspace=nullptr;int workspaceCapacity=48;
    ugkwp::gaswall::WallStatus* failureStatus=nullptr;
};
struct InterfaceGasClosure {
    GasPrimitive trace{};Vec3 traction{};Real conductiveHeatFlux=0;
    ugkwp::gaswall::WallOutput<Real,Ns> layer{};
};
CHMT_HD inline bool evaluateInterfaceGasClosure(const GasPrimitive& bulk,const GasGradient& inputGradient,
    Real temperature,Real distance,Vec3 velocity,Vec3 normal,Real normalSpeed,
    const Real* species,Real fluxTolerance,const PhysicsConfig& p,
    const GasWallClosureContext& context,InterfaceGasClosure& output,bool* numericalFailure=nullptr) {
    InterfaceGasClosure result;
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        if((p.material.gasContactResistance!=0))return false;
        const GasThermoAdapter<Ns> thermo(p.species);
        ugkwp::gaswall::WallInput<Real,Ns> input;
        input.pressure=context.matchingPressure;input.temperature=temperature;
        input.matchingDistance=context.matchingDistance;input.ownerDistance=context.ownerDistance;
        input.matching=context.matching;input.quadrature=context.quadrature;
        input.model.thermo=thermo.view();
        input.diagnostics=p.diagnostics;input.model.thermo.diagnostics=p.diagnostics;
        Real elements[Ns*Ne]{};input.model.thermo.elementCount=p.nElements;input.model.thermo.elementComposition=elements;
        for(int s=0;s<Ns;++s)for(int e=0;e<p.nElements;++e)elements[s*p.nElements+e]=p.species[s].element[e]*thermo.data[s].molarMass;
        input.model.mechanism=context.mechanism;input.model.mode=p.gasMode;
        input.model.viscosity=p.gasViscosity;input.model.conductivity=p.gasConductivity;
        auto& coefficients=input.model.sstCoefficients;const auto& sst=p.sst;
        coefficients.a1=sst.a1;coefficients.betaStar=sst.betaStar;coefficients.beta1=sst.beta1;coefficients.beta2=sst.beta2;
        coefficients.gamma1=sst.gamma1;coefficients.gamma2=sst.gamma2;coefficients.alphaK1=sst.sigmaK1;coefficients.alphaK2=sst.sigmaK2;
        coefficients.alphaOmega1=sst.sigmaOmega1;coefficients.alphaOmega2=sst.sigmaOmega2;coefficients.c1=sst.productionLimit;
        velocity=velocity-normal*dot(velocity,normal)+normal*normalSpeed;
        input.velocity[0]=velocity.x;input.velocity[1]=velocity.y;input.velocity[2]=velocity.z;
        input.normal[0]=normal.x;input.normal[1]=normal.y;input.normal[2]=normal.z;
        for(int s=0;s<Ns;++s){input.massFlux[s]=species[s];input.model.diffusivity[s]=p.gasDiffusivity[s];}
        auto config=p.wallModel;config.enableSst=p.enableSst;
        config.turbulentPrandtl=p.sst.turbulentPrandtl;config.turbulentSchmidt=p.sst.turbulentSchmidt;
        ugkwp::gaswall::WallStatus status;
        bool accepted=false;
        if(config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport)
            accepted=ugkwp::gaswall::evaluateConstantTransportWall(input,config,result.layer,status);
        else if(context.workspace){
            if(context.workspaceCapacity==24)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,24>*>(context.workspace),result.layer,status);
            else if(context.workspaceCapacity==48)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,48>*>(context.workspace),result.layer,status);
            else if(context.workspaceCapacity==128)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,128>*>(context.workspace),result.layer,status);
        }
        if(!accepted){if(numericalFailure)*numericalFailure=false;if(context.failureStatus)*context.failureStatus=status;return false;}
        Real mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!makeInterfacePrimitive(result.layer.traceMassFraction,temperature,input.pressure,velocity,normal,
            normalSpeed,mass,p,result.trace,numericalFailure))return false;
        result.traction={result.layer.traction[0],result.layer.traction[1],result.layer.traction[2]};
        result.conductiveHeatFlux=result.layer.conductiveHeatFlux;
    }else{
        // The previous low-Re path stays bitwise unchanged. Ordinary wall
        // functions have no CHMT mass-transfer extension.
        if((p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe))return false;
        Real Y[Ns]{},mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!interfaceComposition(species,bulk.Y,bulk.pressure,temperature,distance,p,fluxTolerance,Y)
            ||!makeInterfacePrimitive(Y,temperature,bulk.pressure,velocity,normal,normalSpeed,mass,p,result.trace,numericalFailure))return false;
        GasGradient gradient=inputGradient;const Vec3 difference=bulk.velocity-result.trace.velocity;
        gradient.velocity[0]+=normal*(difference.x/distance-dot(gradient.velocity[0],normal));
        gradient.velocity[1]+=normal*(difference.y/distance-dot(gradient.velocity[1],normal));
        gradient.velocity[2]+=normal*(difference.z/distance-dot(gradient.velocity[2],normal));
        result.traction=viscousTraction(gradient,normal,p.gasViscosity,0);
        const Real conductance=p.gasConductivity>0?1/(distance/p.gasConductivity+p.material.gasContactResistance):0;
        result.conductiveHeatFlux=conductance*(temperature-bulk.temperature);
    }
    output=result;return true;
}
CHMT_HD inline bool hertzKnudsenFlux(const FilmQ& film, const Real* gasY, Real pressure,
    Real T, const PhysicsConfig& p, Real* flux) {
    if ((film.mass<=0 || T<=0 || !finite(T))) return false;
    Real molecularSum=0;
    for (int s=0; s<Ns; ++s) molecularSum+=film.species[s]*p.species[s].R;
    if ((!(molecularSum>0))) return false;
    for (int s=0; s<Ns; ++s) {
        const Real alpha=physicalFraction(p,p.material.evaporationAccommodation[s],s);
        const Real activity=film.species[s]*p.species[s].R/molecularSum;
        const Real saturation=p.material.saturationPressureScale[s]
            *::exp(-p.material.saturationTemperature[s]/T);
        flux[s]=alpha*(activity*saturation-interfacePartialPressure(pressure,gasY,p,s))
            /::sqrt(2*3.14159265358979323846*p.species[s].R*T);
        if ((!finite(flux[s]))) return false;
    }
    return true;
}

struct SurfacePhysicsInput {
    GasWallClosureContext wallContext{};
    GasPrimitive bulk{};
    GasGradient gradient{};
    GasQ gasInventory{};
    SolidQ solid{}, solidBase{}, solidLocalRate{};
    FilmQ film{}, filmBase{}, filmLocalRate{};
    FilmAux aux{};
    bool hasSolid=false, hasFilm=false, useAcceptedFlux=false;
    Real solidVolume=0, area=0, gasDistance=0, solidDistance=0, dt=0;
    Vec3 normal{}, baseVelocity{};
    Real gasArea=0, solidArea=0;
    Vec3 gasNormal{}, solidNormal{}; // both point from material toward gas.
    bool useSweptGeometry=false;
    Real gasSweptRate=0, solidSweptRate=0; // physical normal swept volume rates, m^3/s.
    Real filmTransportMassRate=0, sideMeshVolumeRate=0;
    Real acceptedGasSpecies[Ns]{}, acceptedPoreSpecies[Ns]{}, acceptedPoreSweep[Ns]{}, acceptedCondensed[Nc]{};
    Real acceptedPhaseMass=0;
};
struct SurfacePhysicsResult {
    GasPrimitive gasState{};
    Real gasSpecies[Ns]{}, poreSpecies[Ns]{}, poreDarcy[Ns]{}, poreSweep[Ns]{}, condensed[Nc]{}, phaseMass=0;
    Real gasEnergy=0, poreEnergy=0, phaseEnergy=0;
    Real gasAdvective=0, gasConductive=0, gasPressureWork=0, gasViscousWork=0;
    Real phaseAdvective=0, phaseConductive=0, phasePressureWork=0, phaseViscousWork=0;
    Real topKineticAdvection=0, bottomKineticAdvection=0;
    Real topTemperature=0, bottomTemperature=0, solidSpeed=0, topSpeed=0;
    Real radiationFlux=0, thermalResidual=0;
    Real gasWallSpeed=0, solidWallSpeed=0;
    Real liquidTopNormal=0, liquidBottomNormal=0; // components along reference normal.
    Real solidRecessionResidual=0, filmVolumeResidual=0; // m^3/s, not heat sources.
    Vec3 shear{}, gasShear{}, traction{};
    bool wet=false;
};
CHMT_HD inline Real phaseRateFromHeat(Real power, Real latent, Real densityJump) {
    if ((!(latent>0) || !finite(power))) return undefinedValue();
    Real low=0, high=absValue(power)/latent;
    for (int iteration=0; iteration<80; ++iteration) {
        const Real rate=0.5*(low+high);
        const Real value=rate*(latent+0.5*rate*rate*densityJump*densityJump);
        if (value>absValue(power)) high=rate; else low=rate;
    }
    return (power>=0 ? 1 : -1)*0.5*(low+high);
}
// Evaluate the product in logarithmic scale so individually overflowing powers
// cannot poison a finite rate or trigger OpenFOAM's enabled overflow trap.
CHMT_HD inline Real surfaceDonorFraction(Real available,Real scale,Real loss) {
    const Real withdrawal=auditProduct(scale,loss);
    if(withdrawal<=available)return 1;
    return scale>=1?(available/scale)/loss:(available/loss)/scale;
}
CHMT_HD inline Real surfaceArrheniusRate(const Reaction& r, const Real* gasOrder,
    const SurfacePhysicsInput& in, const GasPrimitive& gas, Real T) {
    if(!finite(T)||T<=0||!finite(r.A)||r.A<0||!finite(r.temperaturePower)
        ||!finite(r.activationEnergy)||r.activationEnergy<0)return undefinedValue();
    bool zero=r.A==0;Real concentrationLog=0;
    for(int c=0;c<Nc;++c){const Real order=r.order[c];
        if(!finite(order)||order<0)return undefinedValue();
        if(order==0)continue;
        const Real mass=in.solid.condensed[c];
        if(!finite(mass)||mass<0||!finite(in.solidVolume)||in.solidVolume<=0)return undefinedValue();
        if(mass==0)zero=true;
        else concentrationLog=auditAdd(concentrationLog,auditProduct(order,::log(mass)-::log(in.solidVolume)));
    }
    for(int species=0;species<Ns;++species){const Real order=gasOrder[species];
        if(!finite(order)||order<0)return undefinedValue();
        if(order==0)continue;
        if(!finite(gas.rho)||gas.rho<0||!finite(gas.Y[species])||gas.Y[species]<0)return undefinedValue();
        if(gas.rho==0||gas.Y[species]==0)zero=true;
        else concentrationLog=auditAdd(concentrationLog,auditProduct(order,::log(gas.rho)+::log(gas.Y[species])));
    }
    if(zero)return 0;
    const Real activation=(T<1&&r.activationEnergy>std::numeric_limits<Real>::max()*T)
        ?std::numeric_limits<Real>::infinity():(r.activationEnergy/T)/universalGasConstant;
    Real logarithm=auditAdd(::log(r.A),auditProduct(r.temperaturePower,::log(T)));
    logarithm=auditAdd(auditDifference(logarithm,activation),concentrationLog);
    if(logarithm!=logarithm)return undefinedValue();
    const Real maximumLog=::log(std::numeric_limits<Real>::max());
    if(logarithm>maximumLog)return std::numeric_limits<Real>::infinity();
    if(logarithm<::log(std::numeric_limits<Real>::denorm_min())-::log(Real(2)))return 0;
    return ::exp(logarithm);
}
CHMT_HD inline bool evaluateSurfaceTemperature(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, const MaterialPrimitive& material, Real filmT, Real topT,
    bool birth, SurfacePhysicsResult& out,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if((p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer))return false;
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:in.bulk.pressure;
    if((!finite(mechanicalPressure)||mechanicalPressure<=0))return interfaceFailure(p,3844,mechanicalPressure);
    SurfacePhysicsResult result;
    const int phase=p.material.phaseCondensed;
    const Real gasRatio=in.gasArea/in.area;
    const Real solidRatio=in.hasSolid ? in.solidArea/in.area : 0;
    const Real gasProjection=dot(in.normal,in.gasNormal);
    const Real solidProjection=in.hasSolid ? dot(in.normal,in.solidNormal) : 1;
    result.wet=in.hasFilm && (in.film.mass>0 || birth);
    result.topTemperature=topT;
    const Real gasConductance=p.gasConductivity>0
        ? 1/(in.gasDistance/p.gasConductivity+p.material.gasContactResistance) : 0;
    const Real solidConductance=in.hasSolid && material.conductivity>0
        ? 1/(in.solidDistance/material.conductivity+p.material.solidContactResistance) : 0;
    const Real thickness=in.hasFilm && in.film.mass>0 ? in.film.mass/(p.liquid.rho*in.area) : 0;
    const Real liquidConductance=thickness>0 && p.liquid.conductivity>0
        ? 2*p.liquid.conductivity/thickness : 0;
    result.radiationFlux=gasRatio*grayRadiation(topT,p);
    Real bottomT=in.hasSolid ? material.temperature : filmT;
    Real phaseRate=0;
    if (result.wet && in.hasSolid) {
        if (p.material.enableMelting) {
            if ((phase<0 || phase>=Nc)) return false;
            bottomT=p.meltTemperature;
            const Real latent=liquidH(p,bottomT,in.bulk.pressure)
                -condensedE(p.condensed[phase],bottomT)-in.bulk.pressure/p.condensed[phase].rho;
            Real gasIncomingHeat=gasRatio*gasConductance*(in.bulk.temperature-bottomT);
            if(birth&&p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
                Real zeroFlux[Ns]{};InterfaceGasClosure birthClosure;
                if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,bottomT,in.gasDistance,in.baseVelocity,
                    in.gasNormal,0,zeroFlux,p.tolerances.absoluteMass,p,in.wallContext,birthClosure,numericalFailure))return false;
                gasIncomingHeat=-gasRatio*birthClosure.conductiveHeatFlux;
            }
            const Real heat=birth
                ? solidConductance*(material.temperature-bottomT)
                    +(gasIncomingHeat+result.radiationFlux)/solidRatio
                : solidConductance*(material.temperature-bottomT)
                    +solidProjection*liquidConductance*(filmT-bottomT);
            phaseRate=solidRatio*phaseRateFromHeat(heat,latent,1/p.liquid.rho-1/p.condensed[phase].rho);
            if ((!finite(phaseRate))) return interfaceFailure(p,3846,phaseRate,phase,ugkp::diagnostics::Code::NonFinite);
            if ((birth && phaseRate<0)) return false;
        } else if (solidConductance+solidProjection*liquidConductance>0) {
            bottomT=(solidConductance*material.temperature+solidProjection*liquidConductance*filmT)
                /(solidConductance+solidProjection*liquidConductance);
        }
    }
    if (in.useAcceptedFlux) phaseRate=in.acceptedPhaseMass;
    result.bottomTemperature=bottomT;
    Real Y[Ns]{};
    for (int s=0; s<Ns; ++s) Y[s]=in.bulk.Y[s];
    bool compositionConverged=false;Real compositionChange=0;
    InterfaceGasClosure wallClosure;
    // This cache lives only inside this fixed-temperature evaluation. The bulk,
    // model, geometry and matching state cannot change during this loop.
    bool wallClosureReady=false;
    Real wallClosureSpecies[Ns]{},wallClosureSpeed=0;
    Vec3 wallClosureVelocity{};
    for (int iteration=0; iteration<60; ++iteration) {
        SurfacePhysicsResult trial;
        trial.wet=result.wet;
        trial.phaseMass=phaseRate;
        GasPrimitive provisional;
        if (!makeInterfacePrimitive(Y,topT,in.bulk.pressure,{},in.gasNormal,0,0,p,provisional,numericalFailure)) return false;
        if (in.useAcceptedFlux) {
            for (int s=0; s<Ns; ++s) {
                trial.gasSpecies[s]=in.acceptedGasSpecies[s];
                trial.poreSpecies[s]=in.acceptedPoreSpecies[s];
                trial.poreSweep[s]=in.acceptedPoreSweep[s];
                trial.poreDarcy[s]=trial.poreSpecies[s]-trial.poreSweep[s];
            }
            for (int c=0; c<Nc; ++c) trial.condensed[c]=in.acceptedCondensed[c];
        } else {
            if (result.wet && p.material.evaporation==EvaporationLaw::HertzKnudsen && !birth) {
                if (!hertzKnudsenFlux(in.film,Y,in.bulk.pressure,topT,p,trial.gasSpecies)) return false;
                for (int species=0; species<Ns; ++species) trial.gasSpecies[species]*=gasRatio;
            }
            if (!result.wet && p.material.enableSurfaceReactions) {
                if (!in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
                for (int r=0; r<p.material.nSurfaceReactions; ++r) {
                    const Reaction& reaction=p.material.surfaceReactions[r];
                    const Real rate=solidRatio*surfaceArrheniusRate(
                        reaction,p.material.surfaceGasOrder[r],in,provisional,topT);
                    if ((!finite(rate) || rate<0)) return interfaceFailure(p,3845,rate,r);
                    for (int c=0; c<Nc; ++c) trial.condensed[c]-=rate*reaction.condensedNu[c];
                    for (int s=0; s<Ns; ++s) trial.gasSpecies[s]+=rate*reaction.gasNu[s];
                }
            }
        }
        Real consumedVolume=0;
        for (int c=0; c<Nc; ++c) if (trial.condensed[c]!=0) {
            if (!in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
            if(!validCondensed(p.condensed[c]))return false;
            consumedVolume+=trial.condensed[c]/p.condensed[c].rho;
        }
        if (trial.phaseMass!=0) {
            if (!in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
            if((phase<0 || phase>=Nc || !validCondensed(p.condensed[phase])))return false;
            consumedVolume+=trial.phaseMass/p.condensed[phase].rho;
        }
        const Real packing=in.hasSolid ? 1-material.porosity : 1;
        trial.solidSpeed=in.hasSolid ? -consumedVolume/(solidRatio*packing) : 0;
        trial.solidWallSpeed=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate/in.solidArea : trial.solidSpeed;
        if (!in.useAcceptedFlux && in.hasSolid) {
            Real superficial=0;
            if (p.material.enablePoreOutflow && p.permeability>0) {
                superficial=p.permeability/p.poreViscosity
                    *((material.pressure-in.bulk.pressure)/in.solidDistance
                        +material.poreDensity*dot(p.gravity,in.solidNormal));
            }
            // Recession exposes the pore volume even if permeability is zero.
            const Real poreSweep=-material.porosity*trial.solidWallSpeed;
            for (int s=0; s<Ns; ++s) {
                const Real poreDensity=material.porosity>0
                    ? in.solid.pore[s]/(material.porosity*in.solidVolume) : 0;
                trial.poreDarcy[s]=solidRatio*(superficial>=0 ? superficial*poreDensity
                    : superficial*provisional.rho*Y[s]);
                trial.poreSweep[s]=solidRatio*(poreSweep>=0 ? poreSweep*poreDensity
                    : poreSweep*provisional.rho*Y[s]);
                trial.poreSpecies[s]=trial.poreDarcy[s]+trial.poreSweep[s];
                // The wet model needs a pore/film transfer operator only when
                // actual species mass crosses this interface. A geometric pore
                // sweep over an empty donor carries exactly zero mass/energy.
                // Check the separate paths: counterflow can cancel net mass
                // while still transporting donor-dependent energy.
                if(result.wet&&(trial.poreDarcy[s]!=0||trial.poreSweep[s]!=0))
                    return interfaceFailure(p,3870,trial.poreDarcy[s]!=0?trial.poreDarcy[s]:trial.poreSweep[s],s,
                        ugkp::diagnostics::Code::MissingOperator,trial.poreDarcy[s]!=0?trial.poreSweep[s]:trial.poreDarcy[s]);
            }
        }
        Real factor=1;
        if (!in.useAcceptedFlux && in.dt>0) {
            const Real scale=in.area*in.dt;
            for (int c=0; c<Nc; ++c) {
                const Real loss=trial.condensed[c]+(c==phase ? trial.phaseMass : 0);
                if (loss>0) {
                    const Real available=in.solidBase.condensed[c]+in.dt*in.solidLocalRate.condensed[c];
                    factor=minValue(factor,surfaceDonorFraction(available,scale,loss));
                }
            }
            for (int s=0; s<Ns; ++s) {
                if (trial.poreSpecies[s]>0) {
                    const Real available=in.solidBase.pore[s]+in.dt*in.solidLocalRate.pore[s];
                    factor=minValue(factor,surfaceDonorFraction(available,scale,trial.poreSpecies[s]));
                }
                const Real gasLoss=-trial.gasSpecies[s]-trial.poreSpecies[s];
                if (gasLoss>0) factor=minValue(factor,surfaceDonorFraction(in.gasInventory.species[s],scale,gasLoss));
                if (result.wet) {
                    const Real filmLoss=trial.gasSpecies[s]-trial.phaseMass*physicalFraction(p,p.material.phaseFilmY[s],s);
                    if (filmLoss>0) {
                        const Real available=in.filmBase.species[s]+in.dt*in.filmLocalRate.species[s];
                        factor=minValue(factor,surfaceDonorFraction(available,scale,filmLoss));
                    }
                }
            }
            if ((!finite(factor) || factor<0)) return false;
        }
        // Actual geometric sweep is immutable during this evaluation. If a
        // donor limit would rescale it, the entire geometry/source trial must be
        // recomputed instead of silently changing only its diagnostic mass.
        if (in.useSweptGeometry && factor<1) {
            for (int species=0; species<Ns; ++species) if ((trial.poreSweep[species]!=0)) return false;
        }
        trial.phaseMass*=factor;
        trial.solidSpeed*=factor;
        if (!in.useSweptGeometry) trial.solidWallSpeed=trial.solidSpeed;
        Real J=0, filmLoss=0;
        Real allSpecies[Ns]{};
        for (int c=0; c<Nc; ++c) trial.condensed[c]*=factor;
        for (int s=0; s<Ns; ++s) {
            trial.gasSpecies[s]*=factor;
            trial.poreSpecies[s]*=factor;
            trial.poreDarcy[s]*=factor;
            trial.poreSweep[s]*=factor;
            allSpecies[s]=trial.gasSpecies[s]+trial.poreSpecies[s];
            J+=allSpecies[s]; filmLoss+=trial.gasSpecies[s];
        }
        const Real filmVolumeRate=result.wet
            ? (in.filmTransportMassRate+in.area*(trial.phaseMass-filmLoss))/p.liquid.rho : 0;
        trial.topSpeed=result.wet
            ? (in.solidArea*trial.solidSpeed+filmVolumeRate-in.sideMeshVolumeRate)/in.gasArea
            : trial.solidSpeed;
        trial.gasWallSpeed=in.useSweptGeometry ? in.gasSweptRate/in.gasArea : trial.topSpeed;
        trial.solidRecessionResidual=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate-in.solidArea*trial.solidSpeed : 0;
        trial.filmVolumeResidual=in.useSweptGeometry
            ? filmVolumeRate-(in.gasSweptRate-in.solidSweptRate
                +(result.wet ? in.sideMeshVolumeRate : 0)) : 0;
        for (int species=0; species<Ns; ++species) allSpecies[species]/=gasRatio;
        Real nextY[Ns]{};
        const Real fluxTolerance=in.dt>0
            ? p.tolerances.absoluteMass/(in.dt*in.gasArea) : p.tolerances.absoluteMass;
        if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
            Vec3 velocity=in.baseVelocity;
            if(result.wet){const Vec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
                const Real normalVelocity=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)-dot(tangent,in.gasNormal))/gasProjection;
                velocity=tangent+in.normal*normalVelocity;}
            bool sameClosure=wallClosureReady && trial.gasWallSpeed==wallClosureSpeed
                && velocity.x==wallClosureVelocity.x && velocity.y==wallClosureVelocity.y
                && velocity.z==wallClosureVelocity.z;
            for(int s=0;s<Ns;++s) sameClosure=sameClosure && allSpecies[s]==wallClosureSpecies[s];
            if(!sameClosure){
                if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,topT,in.gasDistance,velocity,in.gasNormal,
                    trial.gasWallSpeed,allSpecies,fluxTolerance,p,in.wallContext,wallClosure,numericalFailure))return false;
                wallClosureReady=true;wallClosureSpeed=trial.gasWallSpeed;wallClosureVelocity=velocity;
                for(int s=0;s<Ns;++s)wallClosureSpecies[s]=allSpecies[s];
            }
            for(int s=0;s<Ns;++s)nextY[s]=wallClosure.trace.Y[s];
        }else if (!interfaceComposition(allSpecies,in.bulk.Y,in.bulk.pressure,topT,in.gasDistance,
            p,fluxTolerance,nextY)) return false;
        Real change=0;
        for (int s=0; s<Ns; ++s) change=maxValue(change,absValue(nextY[s]-Y[s]));
        for (int s=0; s<Ns; ++s) Y[s]=0.5*(Y[s]+nextY[s]);
        result.gasState=provisional;
        for (int s=0; s<Ns; ++s) {
            result.gasSpecies[s]=trial.gasSpecies[s]; result.poreSpecies[s]=trial.poreSpecies[s];
            result.poreDarcy[s]=trial.poreDarcy[s]; result.poreSweep[s]=trial.poreSweep[s];
        }
        for (int c=0; c<Nc; ++c) result.condensed[c]=trial.condensed[c];
        result.phaseMass=trial.phaseMass;
        result.solidSpeed=trial.solidSpeed; result.topSpeed=trial.topSpeed;
        result.solidWallSpeed=trial.solidWallSpeed; result.gasWallSpeed=trial.gasWallSpeed;
        result.solidRecessionResidual=trial.solidRecessionResidual;
        result.filmVolumeResidual=trial.filmVolumeResidual;
        Vec3 liquidTrace=in.baseVelocity;
        if (result.wet) {
            const Vec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
            result.liquidTopNormal=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)
                -dot(tangent,in.gasNormal))/gasProjection;
            liquidTrace=tangent+in.normal*result.liquidTopNormal;
        }
        if (!makeInterfacePrimitive(nextY,topT,in.bulk.pressure,liquidTrace,in.gasNormal,
            trial.gasWallSpeed,J/gasRatio,p,result.gasState,numericalFailure)) return false;
        compositionChange=change;
        if (change<=p.tolerances.relativeMass+1e-12) { compositionConverged=true; break; }
    }
    if (!compositionConverged) reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3803,compositionChange,p.tolerances.relativeMass+1e-12);
    Vec3 liquidTop{};
    if (result.wet) {
        liquidTop=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal)
            +in.normal*result.liquidTopNormal;
    }
    GasGradient gradient=in.gradient;
    const Vec3 difference=in.bulk.velocity-result.gasState.velocity;
    gradient.velocity[0]+=in.gasNormal*(difference.x/in.gasDistance-dot(gradient.velocity[0],in.gasNormal));
    gradient.velocity[1]+=in.gasNormal*(difference.y/in.gasDistance-dot(gradient.velocity[1],in.gasNormal));
    gradient.velocity[2]+=in.gasNormal*(difference.z/in.gasDistance-dot(gradient.velocity[2],in.gasNormal));
    // Physical low-Re wall: molecular stress only, no owner nut or k stress.
    result.traction=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.traction:viscousTraction(gradient,in.gasNormal,p.gasViscosity,0);
    result.gasShear=result.traction-in.gasNormal*dot(result.traction,in.gasNormal);
    result.shear=(result.gasShear-in.normal*dot(result.gasShear,in.normal))*gasRatio;
    Real primaryJ=0, poreJ=0, poreDonorEnergy=0, solidDonorEnergy=0;
    for (int s=0; s<Ns; ++s) {
        primaryJ+=result.gasSpecies[s]; poreJ+=result.poreSpecies[s];
        result.gasAdvective+=result.gasSpecies[s]*speciesH(p.species[s],topT);
        result.poreEnergy+=result.poreSpecies[s]*speciesH(p.species[s],topT);
        const Real darcyT=result.poreDarcy[s]>=0 ? material.temperature : topT;
        const Real sweepT=result.poreSweep[s]>=0 ? material.temperature : topT;
        poreDonorEnergy+=result.poreDarcy[s]*speciesH(p.species[s],darcyT)
            +result.poreSweep[s]*speciesE(p.species[s],sweepT);
    }
    const Real gasKE=0.5*dot(result.gasState.velocity,result.gasState.velocity);
    result.gasAdvective+=primaryJ*gasKE;
    result.poreEnergy+=poreJ*gasKE;
    result.gasConductive=gasRatio*(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.conductiveHeatFlux:gasConductance*(topT-in.bulk.temperature));
    result.gasPressureWork=gasRatio*mechanicalPressure*result.gasWallSpeed;
    result.gasViscousWork=-gasRatio*dot(result.traction,result.gasState.velocity);
    result.gasEnergy=result.gasAdvective+result.gasConductive+result.gasPressureWork+result.gasViscousWork;
    const Real solidHeat=solidRatio*solidConductance*(material.temperature-bottomT);
    if (result.wet && in.hasSolid) {
        if (result.phaseMass!=0) {
            if ((phase<0 || phase>=Nc || !validCondensed(p.condensed[phase]))) return false;
            result.phaseAdvective=result.phaseMass*(condensedE(p.condensed[phase],bottomT)
                +in.bulk.pressure/p.condensed[phase].rho);
        }
        result.phaseConductive=solidHeat;
        result.phasePressureWork=solidRatio*in.bulk.pressure*result.solidWallSpeed;
        result.phaseViscousWork=0; // stationary rigid skeleton.
        result.phaseEnergy=result.phaseAdvective+result.phaseConductive+result.phasePressureWork;
        const Vec3 baseTangent=in.baseVelocity-in.normal*dot(in.baseVelocity,in.normal);
        result.liquidBottomNormal=(result.solidWallSpeed+result.phaseMass/(solidRatio*p.liquid.rho)
            -dot(baseTangent,in.solidNormal))/solidProjection;
        const Vec3 liquidBottom=baseTangent+in.normal*result.liquidBottomNormal;
        result.bottomKineticAdvection=-0.5*result.phaseMass*dot(liquidBottom,liquidBottom);
    }
    if (result.wet) {
        result.topKineticAdvection=0.5*primaryJ*dot(liquidTop,liquidTop);
        const Real liquidOut=gasRatio*physicalInterfaceEnergy(primaryJ/gasRatio,
            primaryJ/gasRatio*liquidH(p,topT,in.bulk.pressure),liquidTop,mechanicalPressure,
            result.gasWallSpeed,dot(result.gasShear,liquidTop),
            gasProjection*liquidConductance*(filmT-topT));
        result.thermalResidual=birth ? 0 : result.gasEnergy-liquidOut-result.radiationFlux;
    } else {
        for (int c=0; c<Nc; ++c) if (result.condensed[c]!=0) {
            solidDonorEnergy+=result.condensed[c]*condensedE(p.condensed[c],topT);
        }
        result.thermalResidual=result.gasEnergy+result.poreEnergy-solidDonorEnergy-poreDonorEnergy
            -solidRatio*solidConductance*(material.temperature-topT)-result.radiationFlux;
    }
    if ((!finite(result.thermalResidual) || !finite(result.phaseEnergy))) return interfaceFailure(p,3847,result.thermalResidual,-1,ugkp::diagnostics::Code::NonFinite,result.phaseEnergy);
    out=result; return true;
}
struct SurfaceRootTrace {
    Real attempted=0;bool hasAttempt=false,hasCompleted=false;
    SurfacePhysicsResult completedState{};
};
CHMT_HD inline bool surfaceTemperatureConfiguration(const SurfacePhysicsInput& in,const PhysicsConfig& p) {
    for(int species=0;species<Ns;++species)if(!finite(p.species[species].R)||!(p.species[species].R>0)||!validSpecies(p.species[species]))
        return interfaceSpeciesModelFailure(p,(!finite(p.species[species].R)||p.species[species].R<=0)?p.species[species].R:static_cast<Real>(p.species[species].model),species,(!finite(p.species[species].R)||p.species[species].R<=0)?p.species[species].R:Real(0));
    if(p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer)
        return interfaceFailure(p,3862,static_cast<Real>(p.wallModel.family),-1,ugkp::diagnostics::Code::MissingOperator);
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        const auto& config=p.wallModel;const auto& context=in.wallContext;
        if(p.material.gasContactResistance!=0)
            return interfaceFailure(p,3868,p.material.gasContactResistance,-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&p.gasMode!=ugkwp::GasMode::SingleLegacy&&p.gasMode!=ugkwp::GasMode::MixtureFrozen
            &&p.gasMode!=ugkwp::GasMode::MixtureChemistry)
            return interfaceFailure(p,3869,static_cast<Real>(p.gasMode),-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&config.model!=ugkwp::gaswall::BoundaryLayerModel::ReactingSst)
            return interfaceFailure(p,3863,static_cast<Real>(config.model),-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&(!context.workspace||(context.workspaceCapacity!=24&&context.workspaceCapacity!=48&&context.workspaceCapacity!=128)))
            return interfaceFailure(p,3864,context.workspaceCapacity,-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&(config.nodes<2||config.nodes>context.workspaceCapacity))
            return interfaceFailure(p,3865,config.nodes,-1,ugkp::diagnostics::Code::InvalidIndex,context.workspaceCapacity);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&p.gasMode==ugkwp::GasMode::MixtureChemistry){
            const GasThermoAdapter<Ns> thermo(p.species);
            const auto mechanism=ugkwp::validateGasMechanismView(thermo.view(),context.mechanism);
            if(mechanism.code!=ugkwp::ChemistryCode::Success)
                return interfaceFailure(p,3866,mechanism.reaction,-1,ugkp::diagnostics::Code::MissingOperator);
        }
    }
    if(p.material.enableSurfaceReactions&&(p.material.nSurfaceReactions<0||p.material.nSurfaceReactions>Nr))
        return interfaceFailure(p,3867,p.material.nSurfaceReactions,-1,ugkp::diagnostics::Code::InvalidIndex,Nr);
    return true;
}
// The owner copies immutable physics once per root solve. CUDA device atomics
// must never target an automatic/thread-local DeviceContext.
template<bool Track=false>
CHMT_HD inline bool bisectFeasibleSurfaceImpl(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const PhysicsConfig& trialPhysics,const MaterialPrimitive& material, Real filmT, Real lo, Real hi,
    SurfacePhysicsResult left, SurfacePhysicsResult right, SurfacePhysicsResult& out,SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr) {
    if ((left.thermalResidual>0)==(right.thermalResidual>0)
        && left.thermalResidual!=0 && right.thermalResidual!=0) return false;
    if (left.thermalResidual==0) { out=left; return true; }
    if (right.thermalResidual==0) { out=right; return true; }
    for (int iteration=0; iteration<100; ++iteration) {
        const Real T=0.5*(lo+hi);
        SurfacePhysicsResult candidate;
        if(Track){trace->attempted=T;trace->hasAttempt=true;}
        if(!evaluateSurfaceTemperature(in,trialPhysics,material,filmT,T,false,candidate,numericalFailure))return false;
        if(Track){trace->completedState=candidate;trace->hasCompleted=true;}
        const Real scale=auditAdd(auditAdd(absValue(candidate.gasEnergy),absValue(candidate.poreEnergy)),
            auditAdd(absValue(candidate.phaseEnergy),absValue(candidate.radiationFlux)));
        const Real relativeFluxTolerance=p.tolerances.relativeEnergy==0?Real(0):
            (scale==std::numeric_limits<Real>::infinity()?scale:auditProduct(p.tolerances.relativeEnergy,scale));
        const Real fluxTolerance=auditAdd(auditQuotientProduct(p.tolerances.absoluteEnergy,
            in.area,(in.dt>0?in.dt:Real(1))),relativeFluxTolerance);
        const Real temperatureTolerance=auditTolerance(p.tolerances.absoluteTemperature,p.tolerances.relativeTemperature,T);
        if (absValue(candidate.thermalResidual)<=fluxTolerance
            && (hi-lo<=temperatureTolerance || candidate.thermalResidual==0)) {
            out=candidate; return true;
        }
        if (T==lo || T==hi || iteration==99) {
            reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3804,
                candidate.thermalResidual,fluxTolerance);
            out=candidate;return true;
        }
        if ((candidate.thermalResidual>0)==(left.thermalResidual>0)) { lo=T; left=candidate; }
        else { hi=T; right=candidate; }
    }
    return false;
}
template<bool Track=false>
CHMT_HD inline bool bisectFeasibleSurface(const SurfacePhysicsInput& in,const PhysicsConfig& p,
    const MaterialPrimitive& material,Real filmT,Real lo,Real hi,
    SurfacePhysicsResult left,SurfacePhysicsResult right,SurfacePhysicsResult& out,
    SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr) {
    if(!surfaceTemperatureConfiguration(in,p))return false;
    auto trialPhysics=p;
    trialPhysics.diagnostics=nullptr;trialPhysics.tolerances.diagnostics=nullptr;
    auto trialInput=in;trialInput.wallContext.failureStatus=nullptr;
    SurfacePhysicsResult candidate;
    if(!bisectFeasibleSurfaceImpl<Track>(trialInput,p,trialPhysics,material,filmT,lo,hi,left,right,candidate,trace,numericalFailure))return false;
    return evaluateSurfaceTemperature(in,p,material,filmT,candidate.topTemperature,false,out,numericalFailure);
}
template<bool Track=false>
CHMT_HD inline bool findFeasibleSurfaceRoot(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const MaterialPrimitive& material, Real filmT, Real lo, Real hi, SurfacePhysicsResult& out,SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr) {
    if (!(hi>lo)) return false;
    const Real preferred=maxValue(lo,minValue(hi,in.hasFilm && in.film.mass>0 ? filmT : material.temperature));
    if(!surfaceTemperatureConfiguration(in,p))return false;
    auto trialPhysics=p;
    trialPhysics.diagnostics=nullptr;trialPhysics.tolerances.diagnostics=nullptr;
    auto trialInput=in;trialInput.wallContext.failureStatus=nullptr;
    SurfacePhysicsResult seed;
    if(Track){trace->attempted=preferred;trace->hasAttempt=true;}
    const bool seedValid=evaluateSurfaceTemperature(trialInput,trialPhysics,material,filmT,preferred,false,seed,numericalFailure);
    if(Track&&seedValid){trace->completedState=seed;trace->hasCompleted=true;}
    if (seedValid&&seed.thermalResidual==0) { out=seed; return true; }
    SurfacePhysicsResult last=seed;bool hasValid=seedValid;
    // Invalid samples split feasible components; never bridge an observed hole.
    // Refine bracket discovery without changing the physical time interval.
    for (int subdivisions=32; subdivisions<=512; subdivisions*=4) {
        SurfacePhysicsResult previous;
        Real previousT=lo;
        bool previousValid=false;
        for (int i=0; i<=subdivisions; ++i) {
            const Real gridT=lo+(hi-lo)*static_cast<Real>(i)/subdivisions;
            const bool insert=i>0 && preferred>lo+(hi-lo)*(i-1)/subdivisions && preferred<gridT;
            for (int part=0; part<(insert ? 2 : 1); ++part) {
                const Real T=insert && part==0 ? preferred : gridT;
                SurfacePhysicsResult current;
                if(Track){trace->attempted=T;trace->hasAttempt=true;}
                const bool valid=evaluateSurfaceTemperature(trialInput,trialPhysics,material,filmT,T,false,current,numericalFailure);
                            if(!valid){previousValid=false;continue;}
                last=current;hasValid=true;
                if(Track){trace->completedState=current;trace->hasCompleted=true;}
                if (current.thermalResidual==0) { out=current; return true; }
                if (valid && previousValid
                    && (current.thermalResidual>0)!=(previous.thermalResidual>0)) {
                    SurfacePhysicsResult root;
                    if(!bisectFeasibleSurfaceImpl<Track>(trialInput,p,trialPhysics,material,filmT,previousT,T,previous,current,root,trace,numericalFailure)){
                        if(ugkp::diagnostics::hasFatal(p.diagnostics))return false;
                        previousValid=false;continue;}
                    out=root;return true;
                }
                previousValid=valid;
                if (valid) { previous=current; previousT=T; }
            }
        }
    }
    if(!hasValid){
        // Retain a real failure record when no mathematical state is evaluable.
        SurfacePhysicsResult failed;
        if(!evaluateSurfaceTemperature(in,p,material,filmT,preferred,false,failed,numericalFailure)){
            if(!ugkp::diagnostics::hasFatal(p.diagnostics))interfaceFailure(p,3860,preferred);
            return false;
        }
        return false;
    }
    // No bracket among feasible states is a finite residual budget,
    // so retain the final mathematical iterate, without selecting an earlier state.
    reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3831,last.thermalResidual,0);
    out=last;return true;
}
CHMT_HD inline bool solveSurfaceInterface(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, SurfacePhysicsResult& out, DeviceStatus* status=nullptr,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if ((!(in.area>0 && in.gasArea>0 && in.gasDistance>0) || !finite(in.normal) || !finite(in.gasNormal)
        || mag(in.normal)==0
        || mag(in.gasNormal)==0 || dot(in.normal,in.gasNormal)<=0
        || (in.hasSolid && (mag(in.baseVelocity)!=0 || in.solidDistance<=0 || in.solidArea<=0
            || !finite(in.solidNormal) || mag(in.solidNormal)==0
            || dot(in.normal,in.solidNormal)<=0))
        || (in.useSweptGeometry && (!finite(in.gasSweptRate) || !finite(in.solidSweptRate))))) return interfaceFailure(p,3826,in.area,-1,ugkp::diagnostics::Code::InvalidPhysicalState,in.gasDistance);
    const Real normalResidual=maxValue(absValue(mag(in.normal)-1),maxValue(absValue(mag(in.gasNormal)-1),in.hasSolid?absValue(mag(in.solidNormal)-1):0));
    if(normalResidual>1e-12)reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3827,normalResidual,1e-12);
    MaterialPrimitive material;
    if (in.hasSolid && !recoverMaterial(in.solid,in.solidVolume,p,material,nullptr,-1,numericalFailure)) return false;
    FilmAux filmAux=in.aux;
    const bool wet=in.hasFilm && in.film.mass>0;
    if (wet && !recoverFilm(in.film,in.area,in.aux.pressure,p,filmAux,nullptr,-1)) return false;
    if (!wet && !in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
    if (!wet && (!closeEnough(in.gasArea,in.solidArea,p.tolerances.absoluteGeometry,
        p.tolerances.relativeGeometry) || mag(in.gasNormal-in.solidNormal)>1e-10))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3828,in.gasArea-in.solidArea,mag(in.gasNormal-in.solidNormal));
    if (wet && p.material.enableSurfaceReactions) return interfaceFailure(p,3829,in.film.mass,-1,ugkp::diagnostics::Code::MissingOperator);
    // Pore collapse/bubbling into a film needs an additional mechanical model.
    // The first Stefan phase interface is a dense material phase boundary.
    if (in.hasSolid && p.material.enableMelting && material.porosity>p.tolerances.relativeGeometry) return interfaceFailure(p,3830,material.porosity,-1,ugkp::diagnostics::Code::MissingOperator);
    Real lo=0, hi=std::numeric_limits<Real>::max();
    for (int s=0; s<Ns; ++s) { lo=maxValue(lo,p.species[s].Tmin); hi=minValue(hi,p.species[s].Tmax); }
    if (wet) { lo=maxValue(lo,p.liquid.Tmin); hi=minValue(hi,p.liquid.Tmax); }
    if (in.hasSolid) for (int c=0; c<Nc; ++c) if (in.solid.condensed[c]>0) {
        lo=maxValue(lo,p.condensed[c].Tmin); hi=minValue(hi,p.condensed[c].Tmax);
    }
    const Real filmT=wet ? filmAux.temperature : p.meltTemperature;
    SurfacePhysicsResult candidate;
    if (!findFeasibleSurfaceRoot(in,p,material,filmT,lo,hi,candidate,nullptr,numericalFailure)) return false;
    // Only this final mathematical state publishes diagnostics to the owner.
    SurfacePhysicsResult accepted;
    if(!evaluateSurfaceTemperature(in,p,material,filmT,candidate.topTemperature,false,accepted,numericalFailure))return false;
    candidate=accepted;
    if (!wet && in.hasFilm && p.material.enableMelting && !in.useAcceptedFlux
        && candidate.topTemperature>p.meltTemperature) {
        if (!evaluateSurfaceTemperature(in,p,material,p.meltTemperature,p.meltTemperature,true,candidate,numericalFailure)) return false;
    }
    const bool offsetGeometry=!closeEnough(in.gasArea,in.area,0,p.tolerances.relativeGeometry)
        || mag(in.gasNormal-in.normal)>1e-10
        || (in.hasSolid && (!closeEnough(in.solidArea,in.area,0,p.tolerances.relativeGeometry)
            || mag(in.solidNormal-in.normal)>1e-10));
    const Real rateTolerance=p.tolerances.absoluteGeometry/(in.dt>0 ? in.dt : 1);
    if (offsetGeometry && (absValue(candidate.topSpeed)*in.gasArea>rateTolerance
        || absValue(candidate.solidSpeed)*in.solidArea>rateTolerance
        || (in.useSweptGeometry && (absValue(in.gasSweptRate)>rateTolerance
            || absValue(in.solidSweptRate)>rateTolerance)))) {
        return failStatus(status,ErrorCode::Unsupported,-1,candidate.filmVolumeResidual,ErrorLocation::Face);
    }
    out=candidate; return true;
}
struct SurfacePacketIdentity {
    std::uint64_t step=0, geometry=0, face=0;
    int stage=0, gasCell=-1, solidCell=-1, filmFace=-1;
    Real oldFilmPV=0;
};
CHMT_HD inline ExchangePacket surfacePacketIdentity(ExchangeKind kind, const SurfacePacketIdentity& id) {
    ExchangePacket packet;
    packet.kind=kind; packet.step=id.step; packet.geometry=id.geometry;
    packet.face=id.face; packet.stage=id.stage;
    if (requiredConsumers(kind)&ConsumeGas) packet.gasCell=id.gasCell;
    if (requiredConsumers(kind)&ConsumeSolid) packet.solidCell=id.solidCell;
    if (requiredConsumers(kind)&ConsumeFilm) packet.filmFace=id.filmFace;
    return packet;
}
CHMT_HD inline bool assembleSurfacePackets(const SurfacePhysicsInput& input,
    const SurfacePhysicsResult& result, const SurfacePacketIdentity& identity,
    const PhysicsConfig& p, ExchangePacket* output, DeviceStatus* status=nullptr) {
    const int face=identity.filmFace;
    const Real dt=input.dt;
    const Real scale=input.area*dt;
    const ExchangeKind primaryKind=result.wet ? ExchangeKind::GasFilm : ExchangeKind::GasSolid;
    ExchangePacket primary=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket pore=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket phase=surfacePacketIdentity(primaryKind,identity);
    for (int s=0; s<Ns; ++s) {
        primary.species[s]=scale*result.gasSpecies[s];
        primary.mass+=primary.species[s];
    }
    for (int c=0; c<Nc; ++c) primary.condensed[c]=scale*result.condensed[c];
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?input.wallContext.mechanicalPressure:result.gasState.pressure;
    primary.momentum=result.gasState.velocity*primary.mass
        +(input.gasNormal*mechanicalPressure-result.traction)*(input.gasArea*dt);
    primary.advective=scale*result.gasAdvective;
    primary.conductive=scale*result.gasConductive;
    primary.pressureWork=scale*result.gasPressureWork;
    primary.viscousWork=scale*result.gasViscousWork;
    primary.energy=primary.advective+primary.conductive+primary.pressureWork+primary.viscousWork;
    primary.liquidKineticAdvection=result.wet ? scale*result.topKineticAdvection : 0;
    if (input.hasSolid) {
        pore=surfacePacketIdentity(ExchangeKind::PoreGas,identity);
        for (int s=0; s<Ns; ++s) {
            pore.species[s]=pore.pore[s]=scale*result.poreSpecies[s];
            pore.poreSweep[s]=scale*result.poreSweep[s];
            pore.mass+=pore.species[s];
        }
        pore.momentum=result.gasState.velocity*pore.mass;
        pore.energy=pore.advective=scale*result.poreEnergy;
    }
    if (input.hasSolid && result.wet) {
        phase=surfacePacketIdentity(ExchangeKind::SolidFilm,identity);
        phase.mass=scale*result.phaseMass;
        if (phase.mass!=0) phase.condensed[p.material.phaseCondensed]=phase.mass;
        for (int s=0; s<Ns; ++s) phase.species[s]=phase.mass*physicalFraction(p,p.material.phaseFilmY[s],s);
        phase.advective=scale*result.phaseAdvective;
        phase.conductive=scale*result.phaseConductive;
        phase.pressureWork=scale*result.phasePressureWork;
        phase.viscousWork=scale*result.phaseViscousWork;
        phase.energy=phase.advective+phase.conductive+phase.pressureWork+phase.viscousWork;
        phase.liquidKineticAdvection=scale*result.bottomKineticAdvection;

        // Exact event transfer: the disappearing film hands its actual remaining
        // internal energy to the solid. This is not a correction for R_K.
        FilmQ available=input.filmBase;
        available.mass+=dt*input.filmLocalRate.mass-primary.mass;
        for (int s=0; s<Ns; ++s) {
            available.species[s]+=dt*input.filmLocalRate.species[s]-primary.species[s];
        }
        const Real Uavailable=input.filmBase.enthalpy-identity.oldFilmPV
            +dt*input.filmLocalRate.enthalpy-primary.energy+primary.liquidKineticAdvection
            +scale*result.radiationFlux;
        if (phase.mass<0 && -phase.mass>=available.mass && available.mass>=0) {
            for (int s=0; s<Ns; ++s) {
                const Real expected=available.mass*physicalFraction(p,p.material.phaseFilmY[s],s);
                const Real residual=available.species[s]-expected;
                const Real tolerance=p.tolerances.absoluteMass+p.tolerances.relativeMass
                    *maxValue(absValue(available.species[s]),absValue(expected));
                if(!finite(available.species[s])||!finite(expected)||!finite(residual)||!finite(tolerance))
                    return failStatus(status,ErrorCode::Nonfinite,face,residual,ErrorLocation::Face);
                if (!closeEnough(available.species[s],expected,
                    p.tolerances.absoluteMass,p.tolerances.relativeMass)) {
                    reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,30901,
                        residual,tolerance,residual,s);
                }
            }
            phase.mass=-available.mass;
            for (int c=0; c<Nc; ++c) phase.condensed[c]=c==p.material.phaseCondensed ? phase.mass : 0;
            for (int s=0; s<Ns; ++s) phase.species[s]=-available.species[s];
            phase.energy=-Uavailable-phase.liquidKineticAdvection;
            phase.advective=phase.energy-phase.conductive-phase.pressureWork-phase.viscousWork;
        } else if (available.mass+phase.mass==0) {
            phase.energy=-Uavailable-phase.liquidKineticAdvection;
            phase.advective=phase.energy-phase.conductive-phase.pressureWork-phase.viscousWork;
        }
    }
    DeviceStatus local;
    if (!validatePacketMath(primary,p.tolerances,&local)
        || !validatePacketMath(pore,p.tolerances,&local) || !validatePacketMath(phase,p.tolerances,&local)) {
        return failStatus(status,ErrorCode::Packet,face,local.value,ErrorLocation::Face);
    }
    output[0]=primary; output[1]=pore; output[2]=phase;
    return true;
}
} // namespace chmt
#endif
