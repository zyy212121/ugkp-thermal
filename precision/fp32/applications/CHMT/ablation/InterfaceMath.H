#ifndef CHMT_ABLATION_INTERFACEMATH_H
#define CHMT_ABLATION_INTERFACEMATH_H
#include "gpu/OrderedBisectionProbe.H"
#include "film/FilmMath.H"
#include "../../../common/gasNumerics/OpenFoamViscousFlux.cuh"
#include "../../../common/gasWall/WallModel.H"
#include "../../../common/gasTransport/GasWallCast.H"
namespace chmt {
CHMT_HD inline bool interfaceFailure(const PhysicsConfig& p,unsigned site,PhysicalReal value,int component=-1,
    ugkp::diagnostics::Code code=ugkp::diagnostics::Code::InvalidPhysicalState,PhysicalReal reference=0) {
    using namespace ugkp::diagnostics;
    if(p.diagnostics)recordFatal(p.diagnostics,makeRecord(finite(value)?code:Code::NonFinite,site,38,0,
        EntityKind::Face,-1,component,value,reference,0,Quantity::Accuracy));
    return false;
}



CHMT_HD inline bool surfacePressureFailure(const PhysicsConfig& p,PhysicalReal pressure) {
    return interfaceFailure(p,3844,pressure);
}
CHMT_HD inline void surfaceCompositionWarning(const PhysicsConfig& p,PhysicalReal change,PhysicalReal tolerance) {
    reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3803,change,tolerance);
}
CHMT_HD inline bool surfaceEnergyFailure(const PhysicsConfig& p,PhysicalReal residual,PhysicalReal phaseEnergy) {
    return interfaceFailure(p,3847,residual,-1,ugkp::diagnostics::Code::NonFinite,phaseEnergy);
}
CHMT_HD inline bool surfaceGeometryFailure(const PhysicsConfig& p,PhysicalReal area,PhysicalReal distance) {
    return interfaceFailure(p,3826,area,-1,ugkp::diagnostics::Code::InvalidPhysicalState,distance);
}
CHMT_HD inline void surfaceNormalWarning(const PhysicsConfig& p,PhysicalReal residual) {
    reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3827,residual,PhysicalReal(1e-12));
}
CHMT_HD inline void surfaceAreaWarning(const PhysicsConfig& p,PhysicalReal areaResidual,PhysicalReal normalResidual) {
    reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3828,areaResidual,normalResidual);
}

CHMT_HD inline PhysicalReal interfacePartialPressure(PhysicalReal pressure, const PhysicalReal* Y,
    const PhysicsConfig& p, int species) {
    PhysicalReal R=0;
    for (int s=0; s<Ns; ++s) R+=Y[s]*p.species[s].R;
    return R>0 ? pressure*Y[species]*p.species[species].R/R : undefinedValue();
}
#include "InterfaceCompositionArithmetic.H"
// One producer per diagnostic site keeps component/value provenance while the
// template policy suppresses only private mathematical trial diagnostics.
template<bool Report>
CHMT_HD inline bool compositionInputFailure(const PhysicsConfig& p,PhysicalReal value,int component=-1) {
    if(Report)return interfaceFailure(p,3840,value,component);
    return false;
}
template<bool Report>
CHMT_HD inline bool compositionArithmeticFailure(const PhysicsConfig& p,PhysicalReal value,int component=-1) {
    if(Report)return interfaceFailure(p,3841,value,component,ugkp::diagnostics::Code::NonFinite);
    return false;
}
CHMT_HD inline bool compositionConfigurationFailure(const PhysicsConfig& p,PhysicalReal value,int component=-1,
    ugkp::diagnostics::Code code=ugkp::diagnostics::Code::InvalidPhysicalState) {
    return interfaceFailure(p,3842,value,component,code);
}
// Invocation-local fixed data. No candidate state, density or residual is cached
// here. The owner keeps bulk, fluxes and PhysicsConfig immutable during the solve.
struct CompositionResidualInputs {
    PhysicalReal gasR[Ns]{},totalFlux=0;
};
template<bool Report>
CHMT_HD inline bool prepareCompositionResidual(const PhysicalReal* bulk,const PhysicalReal* Js,
    const PhysicsConfig& p,CompositionResidualInputs& input) {
    for(int i=0;i<Ns;++i){
        input.gasR[i]=p.species[i].R;
        if(!finite(bulk[i]))return compositionInputFailure<Report>(p,bulk[i],i);
        if(!finite(Js[i]))return compositionInputFailure<Report>(p,Js[i],i);
        if(!finite(input.gasR[i])||input.gasR[i]<=0)return compositionInputFailure<Report>(p,input.gasR[i],i);
        if(!finite(p.gasDiffusivity[i])||p.gasDiffusivity[i]<0)
            return compositionInputFailure<Report>(p,p.gasDiffusivity[i],i);
    }
    if(!composition_detail::scaledSum(Js,nullptr,input.totalFlux))
        return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),-1);
    return true;
}
template<bool Report>
CHMT_HD inline bool compositionResidualPolicy(const PhysicalReal* Y,const PhysicalReal* bulk,const PhysicalReal* Js,
    PhysicalReal pressure,PhysicalReal T,PhysicalReal distance,const PhysicsConfig& p,PhysicalReal* residual,PhysicalReal* jacobian,
    const CompositionResidualInputs* prepared=nullptr) {
    using namespace composition_detail;
    PhysicalReal localGasR[Ns]{},R=0,J=prepared?prepared->totalFlux:0,S=0,a[Ns]{},difference[Ns]{};
    const PhysicalReal* gasR=prepared?prepared->gasR:localGasR;
    for(int i=0;i<Ns;++i){
        if(!finite(Y[i]))return compositionInputFailure<Report>(p,Y[i],i);
        if(!prepared){
            localGasR[i]=p.species[i].R;
            if(!finite(bulk[i]))return compositionInputFailure<Report>(p,bulk[i],i);
            if(!finite(Js[i]))return compositionInputFailure<Report>(p,Js[i],i);
            if(!finite(gasR[i])||gasR[i]<=0)return compositionInputFailure<Report>(p,gasR[i],i);
            if(!finite(p.gasDiffusivity[i])||p.gasDiffusivity[i]<0)
                return compositionInputFailure<Report>(p,p.gasDiffusivity[i],i);
        }
    }
    if(!scaledSum(gasR,Y,R)||(!prepared&&!scaledSum(Js,nullptr,J)))
        return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),-1);
    const PhysicalReal denominators[]={R,pressure,T,distance};
    for(int component=0;component<4;++component)
        if(!finite(denominators[component])||!(denominators[component]>0))
            return compositionInputFailure<Report>(p,denominators[component],component);
    DiffusionState diffusionState(pressure,T,distance,R);
    for(int i=0;i<Ns;++i){
        // Evaluate rho*D/distance without an overflowing intermediate product.
        if(!diffusionState.scale(gasMolecularDiffusivity(p,i,T,pressure),a[i])
            ||!sumFinite(Y[i],-bulk[i],difference[i]))
            return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),i);
        PhysicalReal term=0;
        if(!productFinite(a[i],difference[i],term)||!sumFinite(S,term,S))
            return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),i);
    }
    for(int i=0;i<Ns;++i){
        PhysicalReal first=0,second=0,diffusion=0,advective=0,value=0;
        if(!productFinite(a[i],difference[i],first)||!productFinite(Y[i],S,second)
            ||!sumFinite(first,-second,diffusion)||!productFinite(J,Y[i],advective)
            ||!sumFinite(advective,diffusion,value)||!sumFinite(value,-Js[i],residual[i]))
            return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),i);
        if(jacobian)for(int j=0;j<Ns;++j){
            PhysicalReal diagonal=0,off=0,ratio=0,thermo=0;
            if(i==j && (!sumFinite(J,a[i],diagonal)||!sumFinite(diagonal,-S,diagonal)))
                return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),i);
            bool thermoFinite=productFinite(diffusion,gasR[j],thermo)&&quotientFinite(thermo,R,thermo);
            if(!thermoFinite)thermoFinite=quotientFinite(gasR[j],R,ratio)&&productFinite(diffusion,ratio,thermo);
            if(!productFinite(Y[i],a[j],off)||!thermoFinite||!sumFinite(diagonal,-off,value)
                ||!sumFinite(value,-thermo,jacobian[i*Ns+j]))
                return compositionArithmeticFailure<Report>(p,std::numeric_limits<PhysicalReal>::infinity(),i);
        }
    }
    return true;
}
CHMT_HD inline bool compositionResidual(const PhysicalReal* Y,const PhysicalReal* bulk,const PhysicalReal* Js,
    PhysicalReal pressure,PhysicalReal T,PhysicalReal distance,const PhysicsConfig& p,PhysicalReal* residual,PhysicalReal* jacobian) {
    return compositionResidualPolicy<true>(Y,bulk,Js,pressure,T,distance,p,residual,jacobian);
}
#include "InterfaceComposition.H"
CHMT_HD inline bool interfaceComposition(const PhysicalReal* Js,const PhysicalReal* suppliedBulk,
    PhysicalReal pressure,PhysicalReal T,PhysicalReal distance,const PhysicsConfig& p,
    PhysicalReal absoluteFluxTolerance,PhysicalReal* out) {
 PhysicalReal bulk[Ns]{},J=0,scale=0,sum=0,maxD=0;
 CompositionResidualInputs inputs;
 const PhysicalReal denominators[]={pressure,T,distance};
 for(int c=0;c<3;++c)if(!finite(denominators[c])||!(denominators[c]>0))
  return compositionInputFailure<true>(p,denominators[c],c+1);
 for(int i=0;i<Ns;++i){
  inputs.gasR[i]=p.species[i].R;
  bulk[i]=suppliedBulk[i];if(!clipPhysicsMassFraction(p,bulk[i],3805,i))return false;
  if(!finite(Js[i]))return compositionConfigurationFailure(p,Js[i],i);
  if(!finite(p.gasDiffusivity[i])||p.gasDiffusivity[i]<0)
   return compositionConfigurationFailure(p,p.gasDiffusivity[i],i);
  if(!finite(p.species[i].R)||!(p.species[i].R>0))return compositionConfigurationFailure(p,p.species[i].R,i);
  scale=energyArithmetic::maximum(scale,energyArithmetic::absolute(Js[i]));sum+=bulk[i];maxD=energyArithmetic::maximum(maxD,gasMolecularDiffusivity(p,i,T,pressure));
 }
 if(!composition_detail::scaledSum(Js,nullptr,J))
  return compositionConfigurationFailure(p,J,-1,ugkp::diagnostics::Code::NonFinite);
 if(!finite(absoluteFluxTolerance)||absoluteFluxTolerance<0)
  return compositionConfigurationFailure(p,absoluteFluxTolerance);
 if(!finite(p.tolerances.relativeMass)||p.tolerances.relativeMass<0)
  return compositionConfigurationFailure(p,p.tolerances.relativeMass);
 if(!(sum>0))return compositionConfigurationFailure(p,sum);
 if(!closeEnough(sum,1,PhysicalReal(1e-13),p.tolerances.relativeMass))
  reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3806,sum-1,0);
 PhysicalReal relativeScale=0;for(int i=0;i<Ns;++i)if(scale>0)relativeScale+=energyArithmetic::absolute(Js[i]/scale);
 PhysicalReal tolerance=0,relative=0;
 if(!composition_detail::productFinite(p.tolerances.relativeMass,relativeScale,relative)
     ||!composition_detail::productFinite(relative,scale,relative)
     ||!composition_detail::sumFinite(absoluteFluxTolerance,relative,tolerance))tolerance=std::numeric_limits<PhysicalReal>::infinity();
 inputs.totalFlux=J;
 if(maxD==0)return composition_detail::linearComposition(Js,bulk,J,pressure,T,distance,p,tolerance,out,nullptr,&inputs);
 return composition_detail::scalarSolve(Js,bulk,pressure,T,distance,p,tolerance,out,nullptr,&inputs);
}
// Both the primitive builder and the temperature preflight consume this same
// selected species model; retain one source producer for its diagnostic site.
CHMT_HD inline bool interfaceSpeciesModelFailure(const PhysicsConfig& p,PhysicalReal value,
    int component,PhysicalReal reference=0) {
    return interfaceFailure(p,3843,value,component,
        ugkp::diagnostics::Code::MissingOperator,reference);
}
template<class FractionReal>
CHMT_HD inline bool makeInterfacePrimitive(const FractionReal* suppliedY, EnergyReal T, EnergyReal pressure,
    PhysicalVec3 tangentialVelocity, PhysicalVec3 normal, EnergyReal normalSpeed, EnergyReal J,
    const PhysicsConfig& p, GasPrimitive& out,bool* numericalFailure=nullptr) {
    (void)numericalFailure;
    PhysicalReal Y[Ns]{};
    for(int species=0;species<Ns;++species){Y[species]=suppliedY[species];
        if(!clipPhysicsMassFraction(p,Y[species],3809,species))return false;}
    EnergyReal R=0, cv=0, sum=0;
    const GasThermoAdapter<Ns> thermo(p.species);
    const auto thermoView=thermo.view(p.diagnostics);
    for (int s=0; s<Ns; ++s) {
        if(!finite(p.species[s].R)||!(p.species[s].R>0)||!validSpecies(p.species[s]))return interfaceSpeciesModelFailure(p,
            (!finite(p.species[s].R)||p.species[s].R<=0)?p.species[s].R:static_cast<PhysicalReal>(p.species[s].model),s,p.species[s].R);
        if(!finite(Y[s])||Y[s]<0)return interfaceFailure(p,3856,Y[s],s);
        if(!finite(T)||T<=0)return interfaceFailure(p,3857,T,-1);
        R+=EnergyReal(Y[s])*EnergyReal(p.species[s].R);
        cv+=EnergyReal(Y[s])*ugkwp::speciesCv(s,T,thermoView);
        sum+=EnergyReal(Y[s]);
    }
    if(!finite(cv)||cv<=0){using namespace ugkp::diagnostics;
        if(p.diagnostics)recordFatal(p.diagnostics,makeRecord(finite(cv)?Code::InvalidPhysicalState:Code::NonFinite,
            3861,38,0,EntityKind::Face,-1,-1,cv,T,0,Quantity::HeatCapacity));return false;}
    if(!finite(sum)||!finite(pressure)||!finite(R)||pressure<=0||R<=0)return false;
    if(!closeEnough(sum,1,PhysicalReal(1e-13),p.tolerances.relativeMass))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3817,sum-1,0);
    GasPrimitive state;
    state.temperature=T; state.pressure=pressure; state.rho=pressure/(R*T);
    const EnergyReal tangentNormal=surfaceEnergyDot(tangentialVelocity,normal);
    const EnergyReal velocityCorrection=normalSpeed+J/EnergyReal(state.rho)-tangentNormal;
    state.velocity={EnergyReal(tangentialVelocity.x)+EnergyReal(normal.x)*velocityCorrection,
        EnergyReal(tangentialVelocity.y)+EnergyReal(normal.y)*velocityCorrection,
        EnergyReal(tangentialVelocity.z)+EnergyReal(normal.z)*velocityCorrection};
    state.soundSpeed=::sqrtf((EnergyReal(1)+R/cv)*R*T);
    for (int s=0; s<Ns; ++s) state.Y[s]=Y[s];
    if ((!finite(state.velocity) || !finite(state.rho) || state.rho<=0)) return false;
    out=state; return true;
}
// Config/host tables retain their independent input representation. This
// invocation owns all physical coefficient arrays borrowed by the float BVP.
struct SurfaceMolecularTransport {
    ugkwp::SutherlandSpecies<EnergyReal> species[Ns]{};
    ugkwp::PowerLawDiffusivity<EnergyReal> diffusion[Ns]{};
    EnergyReal ratio[Ns*Ns]{},denominator[Ns*Ns]{};
    ugkwp::MolecularViscosityView<EnergyReal> model{};
    CHMT_HD explicit SurfaceMolecularTransport(const PhysicsConfig& p) {
        const auto source=molecularTransportView(p);
        model.model=source.model;model.count=source.count;
        model.compositionSpecies=source.compositionSpecies;model.compositionTRef=source.compositionTRef;
        model.prandtlSpecies=source.prandtlSpecies;model.diffusionCount=source.diffusionCount;
        for(int j=0;j<3;++j){model.muCoefficients[j]=source.muCoefficients[j];model.sCoefficients[j]=source.sCoefficients[j];model.prandtlCoefficients[j]=source.prandtlCoefficients[j];}
        for(int j=0;j<Ns;++j){
            const auto& a=p.gasSutherland[j];auto& b=species[j];
            b.muRef=a.muRef;b.TRef=a.TRef;b.S=a.S;b.molarMass=a.molarMass;
            const auto& d=p.gasPowerLaw[j];auto& e=diffusion[j];
            e.DRef=d.DRef;e.TRef=d.TRef;e.pRef=d.pRef;e.temperatureExponent=d.temperatureExponent;e.pressureExponent=d.pressureExponent;
        }
        for(int j=0;j<Ns*Ns;++j){ratio[j]=p.gasMassRatioQuarter[j];denominator[j]=p.gasWilkeDenominator[j];}
        model.species=species;model.massRatioQuarter=ratio;model.wilkeDenominator=denominator;
        model.diffusion=source.diffusion?diffusion:nullptr;
    }
};
CHMT_HD inline ugkwp::gaswall::WallModelConfig<EnergyReal> surfaceWallModel(const PhysicsConfig& p) {
    auto config=ugkwp::castWallModelConfig<EnergyReal>(p.wallModel);
    config.turbulentPrandtl=p.sst.turbulentPrandtl;config.turbulentSchmidt=p.sst.turbulentSchmidt;
    config.enableSst=p.enableSst;return config;
}
CHMT_HD inline EnergyReal surfaceMolecularViscosity(const PhysicsConfig& p,EnergyReal T,const PhysicalReal* Y) {
    const SurfaceMolecularTransport transport(p);EnergyReal mass[Ns]{};
    for(int j=0;j<Ns;++j)mass[j]=EnergyReal(Y[j]);
    return ugkwp::molecularViscosity<Ns>(transport.model,EnergyReal(p.gasViscosity),T,mass);
}
CHMT_HD inline GasGradient surfaceVelocityGradient(const GasGradient& input,PhysicalVec3 bulk,PhysicalVec3 trace,PhysicalVec3 normal,EnergyReal distance) {
    GasGradient result=input;
    const EnergyReal difference[3]={EnergyReal(bulk.x)-EnergyReal(trace.x),EnergyReal(bulk.y)-EnergyReal(trace.y),EnergyReal(bulk.z)-EnergyReal(trace.z)};
    for(int d=0;d<3;++d){
        const EnergyReal correction=difference[d]/distance-surfaceEnergyDot(input.velocity[d],normal);
        result.velocity[d]={EnergyReal(input.velocity[d].x)+EnergyReal(normal.x)*correction,
            EnergyReal(input.velocity[d].y)+EnergyReal(normal.y)*correction,
            EnergyReal(input.velocity[d].z)+EnergyReal(normal.z)*correction};
    }
    return result;
}
CHMT_HD inline PhysicalVec3 surfaceViscousTraction(const GasGradient& gradient,PhysicalVec3 normal,EnergyReal viscosity) {
    using V=ugkptransport::Vector3T<EnergyReal>;
    const V n{normal.x,normal.y,normal.z};
    const V gx{gradient.velocity[0].x,gradient.velocity[0].y,gradient.velocity[0].z};
    const V gy{gradient.velocity[1].x,gradient.velocity[1].y,gradient.velocity[1].z};
    const V gz{gradient.velocity[2].x,gradient.velocity[2].y,gradient.velocity[2].z};
    const V gn{ugkptransport::dot(gx,n),ugkptransport::dot(gy,n),ugkptransport::dot(gz,n)};
    const auto traction=ugkptransport::openFoamNewtonianTraction(viscosity,n,gn,gx,gy,gz);
    return {traction.x,traction.y,traction.z};
}

// Borrowed stage data. The matching state and geometric quadrature are rebuilt
// against the current stage; canonical mechanism storage remains with the caller.
struct GasWallClosureContext {
    ugkwp::GasMechanismView<EnergyReal,Ns> mechanism{};
    ugkwp::gaswall::WallMatchingState<EnergyReal,Ns> matching{};
    ugkwp::gaswall::WallOwnerQuadrature<EnergyReal> quadrature{};
    // Matching pressure closes EOS/BVP/film thermodynamics. Mechanical pressure
    // is the local gas-face pressure used by momentum and its paired sweep work.
    EnergyReal matchingPressure=0,mechanicalPressure=0,matchingDistance=0,ownerDistance=0;
    void* workspace=nullptr;int workspaceCapacity=48;
    ugkwp::gaswall::WallStatus* failureStatus=nullptr;
    // Optional per-worker publication owned by a cooperative execution team.
    ugkwp::gaswall::WallOutput<EnergyReal,Ns>* teamOutput=nullptr;
    ugkp::diagnostics::DeviceContext* teamDiagnostics=nullptr;
    const PhysicsConfig* quietPhysics=nullptr;
};
struct InterfaceGasClosure {
    GasPrimitive trace{};PhysicalVec3 traction{};EnergyReal conductiveHeatFlux=0;
    ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns> layer{};
};
template<class Team=ugkwp::gaswall::detail::SerialWallTeam>
CHMT_HD inline bool evaluateInterfaceGasClosure(const GasPrimitive& bulk,const GasGradient& inputGradient,
    EnergyReal temperature,EnergyReal distance,PhysicalVec3 velocity,PhysicalVec3 normal,EnergyReal normalSpeed,
    const PhysicalReal* species,PhysicalReal fluxTolerance,const PhysicsConfig& p,
    const GasWallClosureContext& context,InterfaceGasClosure& output,bool* numericalFailure=nullptr,
    const Team& team=Team{}) {
    InterfaceGasClosure result;
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        if((p.material.gasContactResistance!=0))return false;
        const GasThermoAdapter<Ns> thermo(p.species);
        ugkwp::gaswall::WallInput<EnergyReal,Ns> input;
        input.pressure=context.matchingPressure;input.temperature=temperature;
        input.matchingDistance=context.matchingDistance;input.ownerDistance=context.ownerDistance;
        input.matching=context.matching;input.quadrature=context.quadrature;
        input.model.thermo=thermo.view();
        input.diagnostics=p.diagnostics;
        if(context.teamOutput)input.diagnostics=team.broadcast(p.diagnostics!=nullptr)?context.teamDiagnostics:nullptr;
        input.model.thermo.diagnostics=input.diagnostics;
        EnergyReal elements[Ns*Ne]{};input.model.thermo.elementCount=p.nElements;input.model.thermo.elementComposition=elements;
        for(int s=0;s<Ns;++s)for(int e=0;e<p.nElements;++e)elements[s*p.nElements+e]=EnergyReal(p.species[s].element[e])*thermo.data[s].molarMass;
        input.model.mechanism=context.mechanism;input.model.mode=p.gasMode;
        const SurfaceMolecularTransport transport(p);input.model.molecularViscosity=transport.model;input.model.viscosity=p.gasViscosity;input.model.conductivity=p.gasConductivity;input.model.molecularPrandtl=p.gasTransportCoefficients.prandtlSpecies>=0?PhysicalReal(1):PhysicalReal(0);
        auto& coefficients=input.model.sstCoefficients;const auto& sst=p.sst;
        coefficients.a1=sst.a1;coefficients.betaStar=sst.betaStar;coefficients.beta1=sst.beta1;coefficients.beta2=sst.beta2;
        coefficients.gamma1=sst.gamma1;coefficients.gamma2=sst.gamma2;coefficients.alphaK1=sst.sigmaK1;coefficients.alphaK2=sst.sigmaK2;
        coefficients.alphaOmega1=sst.sigmaOmega1;coefficients.alphaOmega2=sst.sigmaOmega2;coefficients.c1=sst.productionLimit;
        const EnergyReal correction=normalSpeed-surfaceEnergyDot(velocity,normal);
        velocity={EnergyReal(velocity.x)+EnergyReal(normal.x)*correction,
            EnergyReal(velocity.y)+EnergyReal(normal.y)*correction,EnergyReal(velocity.z)+EnergyReal(normal.z)*correction};
        input.velocity[0]=velocity.x;input.velocity[1]=velocity.y;input.velocity[2]=velocity.z;
        input.normal[0]=normal.x;input.normal[1]=normal.y;input.normal[2]=normal.z;
        for(int s=0;s<Ns;++s){input.massFlux[s]=species[s];input.model.diffusivity[s]=p.gasDiffusivity[s];}
        auto config=surfaceWallModel(p);
        config.turbulentPrandtl=p.sst.turbulentPrandtl;config.turbulentSchmidt=p.sst.turbulentSchmidt;
        ugkwp::gaswall::WallStatus status;
        ugkwp::gaswall::WallOutput<EnergyReal,Ns> localLayer;
        auto& solvedLayer=context.teamOutput?*context.teamOutput:localLayer;
        bool accepted=false;
        if(config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport)
        {
            if(team.leader())accepted=ugkwp::gaswall::evaluateConstantTransportWall(input,config,solvedLayer,status);
            accepted=team.broadcast(accepted);
        }
        else if(context.workspace){
            if(context.workspaceCapacity==24)accepted=ugkwp::gaswall::evaluateWallModelWithTeam(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,24>*>(context.workspace),solvedLayer,status,team);
            else if(context.workspaceCapacity==48)accepted=ugkwp::gaswall::evaluateWallModelWithTeam(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,48>*>(context.workspace),solvedLayer,status,team);
            else if(context.workspaceCapacity==128)accepted=ugkwp::gaswall::evaluateWallModelWithTeam(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,128>*>(context.workspace),solvedLayer,status,team);
        }
        if(!accepted){if(numericalFailure)*numericalFailure=false;if(team.leader()&&context.failureStatus)*context.failureStatus=status;return false;}
        result.layer=solvedLayer;
        PhysicalReal mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!makeInterfacePrimitive(result.layer.traceMassFraction,temperature,input.pressure,velocity,normal,
            normalSpeed,mass,p,result.trace,numericalFailure))return false;
        result.traction={result.layer.traction[0],result.layer.traction[1],result.layer.traction[2]};
        result.conductiveHeatFlux=result.layer.conductiveHeatFlux;
    }else{
        // The low-Re equations are retained with physical energy arithmetic in
        // EnergyReal. Ordinary wall functions have no CHMT mass-transfer extension.
        if((p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe))return false;
        PhysicalReal Y[Ns]{},mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!interfaceComposition(species,bulk.Y,bulk.pressure,temperature,distance,p,fluxTolerance,Y)
            ||!makeInterfacePrimitive(Y,temperature,bulk.pressure,velocity,normal,normalSpeed,mass,p,result.trace,numericalFailure))return false;
        GasGradient gradient=surfaceVelocityGradient(inputGradient,bulk.velocity,result.trace.velocity,normal,distance);
        result.traction=surfaceViscousTraction(gradient,normal,surfaceMolecularViscosity(p,result.trace.temperature,result.trace.Y));
        const EnergyReal conductivity=gasMolecularConductivity(p,temperature,Y);
        const EnergyReal conductance=conductivity>0?EnergyReal(1)/(distance/conductivity+EnergyReal(p.material.gasContactResistance)):0;
        result.conductiveHeatFlux=conductance*(temperature-EnergyReal(bulk.temperature));
    }
    output=result;team.sync();return true;
}
CHMT_HD inline bool hertzKnudsenFlux(const FilmQ& film, const PhysicalReal* gasY, PhysicalReal pressure,
    PhysicalReal T, const PhysicsConfig& p, PhysicalReal* flux) {
    if ((film.mass<=0 || T<=0 || !finite(T))) return false;
    PhysicalReal molecularSum=0;
    for (int s=0; s<Ns; ++s) molecularSum+=film.species[s]*p.species[s].R;
    if ((!(molecularSum>0))) return false;
    for (int s=0; s<Ns; ++s) {
        const PhysicalReal alpha=physicalFraction(p,p.material.evaporationAccommodation[s],s);
        const PhysicalReal activity=film.species[s]*p.species[s].R/molecularSum;
        const PhysicalReal saturation=p.material.saturationPressureScale[s]
            *std::exp(-p.material.saturationTemperature[s]/T);
        flux[s]=alpha*(activity*saturation-interfacePartialPressure(pressure,gasY,p,s))
            /std::sqrt(2*PhysicalReal(3.14159265358979323846)*p.species[s].R*T);
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
    bool ablating=true; // fixed conjugate faces exchange heat, not material
    PhysicalReal solidVolume=0, area=0, gasDistance=0, solidDistance=0;Real dt=0;
    PhysicalVec3 normal{}, baseVelocity{};
    PhysicalReal gasArea=0, solidArea=0;
    PhysicalVec3 gasNormal{}, solidNormal{}; // both point from material toward gas.
    bool useSweptGeometry=false;
    Real gasSweptRate=0, solidSweptRate=0; // physical normal swept volume rates, m^3/s.
    PhysicalReal filmTransportMassRate=0;Real sideMeshVolumeRate=0;
    PhysicalReal acceptedGasSpecies[Ns]{}, acceptedPoreSpecies[Ns]{}, acceptedPoreSweep[Ns]{}, acceptedCondensed[Nc]{};
    PhysicalReal acceptedPhaseMass=0;
};
struct SurfacePhysicsResult {
    GasPrimitive gasState{};
    PhysicalReal gasSpecies[Ns]{}, poreSpecies[Ns]{}, poreDarcy[Ns]{}, poreSweep[Ns]{}, condensed[Nc]{}, phaseMass=0;
    EnergyReal gasEnergy=0, poreEnergy=0, phaseEnergy=0;
    EnergyReal gasAdvective=0, gasConductive=0, gasPressureWork=0, gasViscousWork=0;
    EnergyReal phaseAdvective=0, phaseConductive=0, phasePressureWork=0, phaseViscousWork=0;
    EnergyReal topKineticAdvection=0, bottomKineticAdvection=0;
    EnergyReal topTemperature=0, bottomTemperature=0;
    PhysicalReal solidSpeed=0, topSpeed=0;
    EnergyReal radiationFlux=0, thermalResidual=0;
    PhysicalReal gasWallSpeed=0, solidWallSpeed=0;
    PhysicalReal liquidTopNormal=0, liquidBottomNormal=0; // components along reference normal.
    Real solidRecessionResidual=0, filmVolumeResidual=0; // m^3/s, not heat sources.
    PhysicalVec3 shear{}, gasShear{}, traction{};
    bool wet=false;
};
CHMT_HD inline EnergyReal phaseRateFromHeat(EnergyReal power, EnergyReal latent, EnergyReal densityJump) {
    if ((!(latent>0) || !finite(power))) return undefinedValue();
    EnergyReal low=0, high=energyArithmetic::absolute(power)/latent;
    for (int iteration=0; iteration<80; ++iteration) {
        const EnergyReal rate=EnergyReal(.5)*(low+high);
        const EnergyReal value=rate*(latent+EnergyReal(.5)*rate*rate*densityJump*densityJump);
        if (value>energyArithmetic::absolute(power)) high=rate; else low=rate;
    }
    return (power>=0 ? 1 : -1)*EnergyReal(.5)*(low+high);
}
// Evaluate the product in logarithmic scale so individually overflowing powers
// cannot poison a finite rate or trigger OpenFOAM's enabled overflow trap.
CHMT_HD inline PhysicalReal surfaceDonorFraction(PhysicalReal available,PhysicalReal scale,PhysicalReal loss) {
    const PhysicalReal withdrawal=surfaceEnergyArithmetic::auditProduct(scale,loss);
    if(withdrawal<=available)return 1;
    return scale>=1?(available/scale)/loss:(available/loss)/scale;
}
CHMT_HD inline PhysicalReal surfaceArrheniusRate(const Reaction& r, const PhysicalReal* gasOrder,
    const SurfacePhysicsInput& in, const GasPrimitive& gas, PhysicalReal T) {
    if(!finite(T)||T<=0||!finite(r.A)||r.A<0||!finite(r.temperaturePower)
        ||!finite(r.activationEnergy)||r.activationEnergy<0)return undefinedValue();
    bool zero=r.A==0;PhysicalReal concentrationLog=0;
    for(int c=0;c<Nc;++c){const PhysicalReal order=r.order[c];
        if(!finite(order)||order<0)return undefinedValue();
        if(order==0)continue;
        const PhysicalReal mass=in.solid.condensed[c];
        if(!finite(mass)||mass<0||!finite(in.solidVolume)||in.solidVolume<=0)return undefinedValue();
        if(mass==0)zero=true;
        else concentrationLog=surfaceEnergyArithmetic::auditAdd(concentrationLog,surfaceEnergyArithmetic::auditProduct(order,std::log(mass)-std::log(in.solidVolume)));
    }
    for(int species=0;species<Ns;++species){const PhysicalReal order=gasOrder[species];
        if(!finite(order)||order<0)return undefinedValue();
        if(order==0)continue;
        if(!finite(gas.rho)||gas.rho<0||!finite(gas.Y[species])||gas.Y[species]<0)return undefinedValue();
        if(gas.rho==0||gas.Y[species]==0)zero=true;
        else concentrationLog=surfaceEnergyArithmetic::auditAdd(concentrationLog,surfaceEnergyArithmetic::auditProduct(order,std::log(gas.rho)+std::log(gas.Y[species])));
    }
    if(zero)return 0;
    const PhysicalReal activation=(T<1&&r.activationEnergy>std::numeric_limits<PhysicalReal>::max()*T)
        ?std::numeric_limits<PhysicalReal>::infinity():(r.activationEnergy/T)/universalGasConstant;
    PhysicalReal logarithm=surfaceEnergyArithmetic::auditAdd(std::log(r.A),surfaceEnergyArithmetic::auditProduct(r.temperaturePower,std::log(T)));
    logarithm=surfaceEnergyArithmetic::auditAdd(surfaceEnergyArithmetic::auditDifference(logarithm,activation),concentrationLog);
    if(logarithm!=logarithm)return undefinedValue();
    const PhysicalReal maximumLog=std::log(std::numeric_limits<PhysicalReal>::max());
    if(logarithm>maximumLog)return std::numeric_limits<PhysicalReal>::infinity();
    if(logarithm<std::log(std::numeric_limits<PhysicalReal>::denorm_min())-std::log(PhysicalReal(2)))return 0;
    return std::exp(logarithm);
}
CHMT_HD inline bool evaluateSurfaceTemperature(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, const MaterialPrimitive& material, EnergyReal filmT, EnergyReal topT,
    bool birth, SurfacePhysicsResult& out,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if((p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer))return false;
    const EnergyReal mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:in.bulk.pressure;
    if((!finite(mechanicalPressure)||mechanicalPressure<=0))return surfacePressureFailure(p,mechanicalPressure);
    SurfacePhysicsResult result;
    const int phase=p.material.phaseCondensed;
    const EnergyReal gasRatio=EnergyReal(in.gasArea)/EnergyReal(in.area);
    const EnergyReal solidRatio=in.hasSolid ? EnergyReal(in.solidArea)/EnergyReal(in.area) : 0;
    const EnergyReal gasProjection=surfaceEnergyDot(in.normal,in.gasNormal);
    const EnergyReal solidProjection=in.hasSolid ? surfaceEnergyDot(in.normal,in.solidNormal) : 1;
    result.wet=in.hasFilm && (in.film.mass>0 || birth);
    result.topTemperature=topT;
    const EnergyReal gasConductivity=gasMolecularConductivity(p,topT,in.bulk.Y);
    const EnergyReal gasConductance=gasConductivity>0
        ? EnergyReal(1)/(EnergyReal(in.gasDistance)/gasConductivity+EnergyReal(p.material.gasContactResistance)) : 0;
    const EnergyReal solidConductance=in.hasSolid && material.conductivity>0
        ? EnergyReal(1)/(EnergyReal(in.solidDistance)/EnergyReal(material.conductivity)+EnergyReal(p.material.solidContactResistance)) : 0;
    const EnergyReal thickness=in.hasFilm && in.film.mass>0 ? EnergyReal(in.film.mass)/(EnergyReal(p.liquid.rho)*EnergyReal(in.area)) : 0;
    const EnergyReal liquidConductance=thickness>0 && p.liquid.conductivity>0
        ? EnergyReal(2)*EnergyReal(p.liquid.conductivity)/thickness : 0;
    result.radiationFlux=gasRatio*grayRadiation(topT,p);
    EnergyReal bottomT=in.hasSolid ? material.temperature : filmT;
    PhysicalReal phaseRate=0;
    if (result.wet && in.hasSolid) {
        if (p.material.enableMelting) {
            if ((phase<0 || phase>=Nc)) return false;
            bottomT=p.meltTemperature;
            const EnergyReal latent=liquidH(p,bottomT,in.bulk.pressure)
                -condensedE(p.condensed[phase],bottomT)-EnergyReal(in.bulk.pressure)/EnergyReal(p.condensed[phase].rho);
            EnergyReal gasIncomingHeat=gasRatio*gasConductance*(EnergyReal(in.bulk.temperature)-bottomT);
            if(birth&&p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
                PhysicalReal zeroFlux[Ns]{};InterfaceGasClosure birthClosure;
                if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,bottomT,in.gasDistance,in.baseVelocity,
                    in.gasNormal,0,zeroFlux,p.tolerances.absoluteMass,p,in.wallContext,birthClosure,numericalFailure))return false;
                gasIncomingHeat=-gasRatio*birthClosure.conductiveHeatFlux;
            }
            const EnergyReal heat=birth
                ? solidConductance*(EnergyReal(material.temperature)-bottomT)
                    +(gasIncomingHeat+result.radiationFlux)/solidRatio
                : solidConductance*(EnergyReal(material.temperature)-bottomT)
                    +solidProjection*liquidConductance*(filmT-bottomT);
            phaseRate=solidRatio*phaseRateFromHeat(heat,latent,EnergyReal(1)/EnergyReal(p.liquid.rho)-EnergyReal(1)/EnergyReal(p.condensed[phase].rho));
            if ((!finite(phaseRate))) return interfaceFailure(p,3846,phaseRate,phase,ugkp::diagnostics::Code::NonFinite);
            if ((birth && phaseRate<0)) return false;
        } else if (solidConductance+solidProjection*liquidConductance>0) {
            bottomT=(solidConductance*EnergyReal(material.temperature)+solidProjection*liquidConductance*filmT)
                /(solidConductance+solidProjection*liquidConductance);
        }
    }
    if (in.useAcceptedFlux) phaseRate=in.acceptedPhaseMass;
    result.bottomTemperature=bottomT;
    PhysicalReal Y[Ns]{};
    for (int s=0; s<Ns; ++s) Y[s]=in.bulk.Y[s];
    bool compositionConverged=false;PhysicalReal compositionChange=0;
    InterfaceGasClosure wallClosure;
    // This cache lives only inside this fixed-temperature evaluation. The bulk,
    // model, geometry and matching state cannot change during this loop.
    bool wallClosureReady=false;
    PhysicalReal wallClosureSpecies[Ns]{},wallClosureSpeed=0;
    PhysicalVec3 wallClosureVelocity{};
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
                    const PhysicalReal rate=solidRatio*surfaceArrheniusRate(
                        reaction,p.material.surfaceGasOrder[r],in,provisional,topT);
                    if ((!finite(rate) || rate<0)) return interfaceFailure(p,3845,rate,r);
                    for (int c=0; c<Nc; ++c) trial.condensed[c]-=rate*reaction.condensedNu[c];
                    for (int s=0; s<Ns; ++s) trial.gasSpecies[s]+=rate*reaction.gasNu[s];
                }
            }
        }
        PhysicalReal consumedVolume=0;
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
        const PhysicalReal packing=in.hasSolid ? 1-material.porosity : 1;
        trial.solidSpeed=in.hasSolid ? -consumedVolume/(solidRatio*packing) : 0;
        trial.solidWallSpeed=in.hasSolid && in.useSweptGeometry
            ? PhysicalReal(in.solidSweptRate)/in.solidArea : trial.solidSpeed;
        if (!in.useAcceptedFlux && in.hasSolid) {
            PhysicalReal superficial=0;
            if (p.material.enablePoreOutflow && p.permeability>0) {
                superficial=p.permeability/p.poreViscosity
                    *((material.pressure-in.bulk.pressure)/in.solidDistance
                        +material.poreDensity*dot(p.gravity,in.solidNormal));
            }
            // Recession exposes the pore volume even if permeability is zero.
            const PhysicalReal poreSweep=-material.porosity*trial.solidWallSpeed;
            for (int s=0; s<Ns; ++s) {
                const PhysicalReal poreDensity=material.porosity>0
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
        PhysicalReal factor=1;
        if (!in.useAcceptedFlux && in.dt>0) {
            const PhysicalReal scale=in.area*PhysicalReal(in.dt);
            for (int c=0; c<Nc; ++c) {
                const PhysicalReal loss=trial.condensed[c]+(c==phase ? trial.phaseMass : 0);
                if (loss>0) {
                    const PhysicalReal available=in.solidBase.condensed[c]+PhysicalReal(in.dt)*in.solidLocalRate.condensed[c];
                    factor=energyArithmetic::minimum(factor,surfaceDonorFraction(available,scale,loss));
                }
            }
            for (int s=0; s<Ns; ++s) {
                if (trial.poreSpecies[s]>0) {
                    const PhysicalReal available=in.solidBase.pore[s]+PhysicalReal(in.dt)*in.solidLocalRate.pore[s];
                    factor=energyArithmetic::minimum(factor,surfaceDonorFraction(available,scale,trial.poreSpecies[s]));
                }
                const PhysicalReal gasLoss=-trial.gasSpecies[s]-trial.poreSpecies[s];
                if (gasLoss>0) factor=energyArithmetic::minimum(factor,surfaceDonorFraction(in.gasInventory.species[s],scale,gasLoss));
                if (result.wet) {
                    const PhysicalReal filmLoss=trial.gasSpecies[s]-trial.phaseMass*physicalFraction(p,p.material.phaseFilmY[s],s);
                    if (filmLoss>0) {
                        const PhysicalReal available=in.filmBase.species[s]+PhysicalReal(in.dt)*in.filmLocalRate.species[s];
                        factor=energyArithmetic::minimum(factor,surfaceDonorFraction(available,scale,filmLoss));
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
        PhysicalReal J=0, filmLoss=0;
        PhysicalReal allSpecies[Ns]{};
        for (int c=0; c<Nc; ++c) trial.condensed[c]*=factor;
        for (int s=0; s<Ns; ++s) {
            trial.gasSpecies[s]*=factor;
            trial.poreSpecies[s]*=factor;
            trial.poreDarcy[s]*=factor;
            trial.poreSweep[s]*=factor;
            allSpecies[s]=trial.gasSpecies[s]+trial.poreSpecies[s];
            J+=allSpecies[s]; filmLoss+=trial.gasSpecies[s];
        }
        const PhysicalReal filmVolumeRate=result.wet
            ? (in.filmTransportMassRate+in.area*(trial.phaseMass-filmLoss))/p.liquid.rho : 0;
        trial.topSpeed=result.wet
            ? (in.solidArea*trial.solidSpeed+filmVolumeRate-PhysicalReal(in.sideMeshVolumeRate))/in.gasArea
            : trial.solidSpeed;
        trial.gasWallSpeed=in.useSweptGeometry ? PhysicalReal(in.gasSweptRate)/in.gasArea : trial.topSpeed;
        trial.solidRecessionResidual=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate-Real(in.solidArea*trial.solidSpeed) : 0;
        trial.filmVolumeResidual=in.useSweptGeometry
            ? Real(filmVolumeRate)-(in.gasSweptRate-in.solidSweptRate
                +(result.wet ? in.sideMeshVolumeRate : 0)) : 0;
        for (int species=0; species<Ns; ++species) allSpecies[species]/=gasRatio;
        PhysicalReal nextY[Ns]{};
        const PhysicalReal fluxTolerance=in.dt>0
            ? p.tolerances.absoluteMass/(PhysicalReal(in.dt)*in.gasArea) : p.tolerances.absoluteMass;
        if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
            PhysicalVec3 velocity=in.baseVelocity;
            if(result.wet){const PhysicalVec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
                const PhysicalReal normalVelocity=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)-dot(tangent,in.gasNormal))/gasProjection;
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
        PhysicalReal change=0;
        for (int s=0; s<Ns; ++s) change=energyArithmetic::maximum(change,energyArithmetic::absolute(nextY[s]-Y[s]));
        for (int s=0; s<Ns; ++s) Y[s]=PhysicalReal(0.5)*(Y[s]+nextY[s]);
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
        PhysicalVec3 liquidTrace=in.baseVelocity;
        if (result.wet) {
            const PhysicalVec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
            result.liquidTopNormal=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)
                -dot(tangent,in.gasNormal))/gasProjection;
            liquidTrace=tangent+in.normal*result.liquidTopNormal;
        }
        if (!makeInterfacePrimitive(nextY,topT,in.bulk.pressure,liquidTrace,in.gasNormal,
            trial.gasWallSpeed,J/gasRatio,p,result.gasState,numericalFailure)) return false;
        compositionChange=change;
        if (change<=p.tolerances.relativeMass+PhysicalReal(1e-12)) { compositionConverged=true; break; }
    }
    if (!compositionConverged) surfaceCompositionWarning(p,compositionChange,p.tolerances.relativeMass+PhysicalReal(1e-12));
    PhysicalVec3 liquidTop{};
    if (result.wet) {
        liquidTop=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal)
            +in.normal*result.liquidTopNormal;
    }
    GasGradient gradient=surfaceVelocityGradient(in.gradient,in.bulk.velocity,result.gasState.velocity,in.gasNormal,EnergyReal(in.gasDistance));
    // Physical low-Re wall: molecular stress only, no owner nut or k stress.
    result.traction=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.traction:surfaceViscousTraction(gradient,in.gasNormal,surfaceMolecularViscosity(p,result.gasState.temperature,result.gasState.Y));
    result.gasShear=result.traction-in.gasNormal*dot(result.traction,in.gasNormal);
    result.shear=(result.gasShear-in.normal*dot(result.gasShear,in.normal))*gasRatio;
    EnergyReal primaryJ=0, poreJ=0, poreDonorEnergy=0, solidDonorEnergy=0;
    for (int s=0; s<Ns; ++s) {
        primaryJ+=EnergyReal(result.gasSpecies[s]); poreJ+=EnergyReal(result.poreSpecies[s]);
        result.gasAdvective+=EnergyReal(result.gasSpecies[s])*speciesH(p.species[s],topT);
        result.poreEnergy+=EnergyReal(result.poreSpecies[s])*speciesH(p.species[s],topT);
        const EnergyReal darcyT=result.poreDarcy[s]>=0 ? material.temperature : topT;
        const EnergyReal sweepT=result.poreSweep[s]>=0 ? material.temperature : topT;
        poreDonorEnergy+=EnergyReal(result.poreDarcy[s])*speciesH(p.species[s],darcyT)
            +EnergyReal(result.poreSweep[s])*speciesE(p.species[s],sweepT);
    }
    const EnergyReal gasKE=EnergyReal(.5)*surfaceEnergyDot(result.gasState.velocity,result.gasState.velocity);
    result.gasAdvective+=primaryJ*gasKE;
    result.poreEnergy+=poreJ*gasKE;
    result.gasConductive=gasRatio*((p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer||p.gasTransportCoefficients.prandtlSpecies>=0)
        ?wallClosure.conductiveHeatFlux:gasConductance*(topT-EnergyReal(in.bulk.temperature)));
    result.gasPressureWork=gasRatio*mechanicalPressure*EnergyReal(result.gasWallSpeed);
    result.gasViscousWork=-gasRatio*surfaceEnergyDot(result.traction,result.gasState.velocity);
    result.gasEnergy=result.gasAdvective+result.gasConductive+result.gasPressureWork+result.gasViscousWork;
    const EnergyReal solidHeat=solidRatio*solidConductance*(EnergyReal(material.temperature)-bottomT);
    if (result.wet && in.hasSolid) {
        if (result.phaseMass!=0) {
            if ((phase<0 || phase>=Nc || !validCondensed(p.condensed[phase]))) return false;
            result.phaseAdvective=EnergyReal(result.phaseMass)*(condensedE(p.condensed[phase],bottomT)
                +EnergyReal(in.bulk.pressure)/EnergyReal(p.condensed[phase].rho));
        }
        result.phaseConductive=solidHeat;
        result.phasePressureWork=solidRatio*EnergyReal(in.bulk.pressure)*EnergyReal(result.solidWallSpeed);
        result.phaseViscousWork=0; // stationary rigid skeleton.
        result.phaseEnergy=result.phaseAdvective+result.phaseConductive+result.phasePressureWork;
        const PhysicalVec3 baseTangent=in.baseVelocity-in.normal*dot(in.baseVelocity,in.normal);
        result.liquidBottomNormal=(result.solidWallSpeed+result.phaseMass/(solidRatio*p.liquid.rho)
            -dot(baseTangent,in.solidNormal))/solidProjection;
        const PhysicalVec3 liquidBottom=baseTangent+in.normal*result.liquidBottomNormal;
        result.bottomKineticAdvection=-EnergyReal(.5)*EnergyReal(result.phaseMass)*surfaceEnergyDot(liquidBottom,liquidBottom);
    }
    if (result.wet) {
        result.topKineticAdvection=EnergyReal(.5)*primaryJ*surfaceEnergyDot(liquidTop,liquidTop);
        const EnergyReal liquidOut=gasRatio*physicalInterfaceEnergy(primaryJ/gasRatio,
            primaryJ/gasRatio*liquidH(p,topT,in.bulk.pressure),liquidTop,mechanicalPressure,
            result.gasWallSpeed,surfaceEnergyDot(result.gasShear,liquidTop),
            gasProjection*liquidConductance*(filmT-topT));
        result.thermalResidual=birth ? 0 : result.gasEnergy-liquidOut-result.radiationFlux;
    } else {
        for (int c=0; c<Nc; ++c) if (result.condensed[c]!=0) {
            solidDonorEnergy+=EnergyReal(result.condensed[c])*condensedE(p.condensed[c],topT);
        }
        result.thermalResidual=result.gasEnergy+result.poreEnergy-solidDonorEnergy-poreDonorEnergy
            -solidRatio*solidConductance*(EnergyReal(material.temperature)-topT)-result.radiationFlux;
    }
    if ((!finite(result.thermalResidual) || !finite(result.phaseEnergy))) return surfaceEnergyFailure(p,result.thermalResidual,result.phaseEnergy);
    out=result; return true;
}


struct LegacySurfaceEvaluation {
    CHMT_HD static bool evaluate(const SurfacePhysicsInput& in,const PhysicsConfig& p,
        const MaterialPrimitive& material,EnergyReal filmT,EnergyReal topT,bool birth,
        SurfacePhysicsResult& out,bool* numericalFailure) {
        return evaluateSurfaceTemperature(in,p,material,filmT,topT,birth,out,numericalFailure);
    }
};
struct SurfaceRootTrace {
    EnergyReal attempted=0;bool hasAttempt=false,hasCompleted=false;
    SurfacePhysicsResult completedState{};
};
CHMT_HD inline bool surfaceTemperatureConfiguration(const SurfacePhysicsInput& in,const PhysicsConfig& p) {
    for(int species=0;species<Ns;++species)if(!finite(p.species[species].R)||!(p.species[species].R>0)||!validSpecies(p.species[species]))
        return interfaceSpeciesModelFailure(p,(!finite(p.species[species].R)||p.species[species].R<=0)?p.species[species].R:static_cast<PhysicalReal>(p.species[species].model),species,(!finite(p.species[species].R)||p.species[species].R<=0)?p.species[species].R:PhysicalReal(0));
    if(p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer)
        return interfaceFailure(p,3862,static_cast<PhysicalReal>(p.wallModel.family),-1,ugkp::diagnostics::Code::MissingOperator);
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        const auto& config=p.wallModel;const auto& context=in.wallContext;
        if(p.material.gasContactResistance!=0)
            return interfaceFailure(p,3868,p.material.gasContactResistance,-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&p.gasMode!=ugkwp::GasMode::SingleLegacy&&p.gasMode!=ugkwp::GasMode::MixtureFrozen
            &&p.gasMode!=ugkwp::GasMode::MixtureChemistry)
            return interfaceFailure(p,3869,static_cast<PhysicalReal>(p.gasMode),-1,ugkp::diagnostics::Code::MissingOperator);
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
            &&config.model!=ugkwp::gaswall::BoundaryLayerModel::ReactingSst
            &&config.model!=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde)
            return interfaceFailure(p,3863,static_cast<PhysicalReal>(config.model),-1,ugkp::diagnostics::Code::MissingOperator);
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
CHMT_HD inline EnergyReal surfaceThermalFluxTolerance(const SurfacePhysicsInput& in,const PhysicsConfig& p,const SurfacePhysicsResult& result) {
    const EnergyReal scale=surfaceEnergyArithmetic::auditAdd(surfaceEnergyArithmetic::auditAdd(energyArithmetic::absolute(result.gasEnergy),energyArithmetic::absolute(result.poreEnergy)),
        surfaceEnergyArithmetic::auditAdd(energyArithmetic::absolute(result.phaseEnergy),energyArithmetic::absolute(result.radiationFlux)));
    const EnergyReal suppliedRelative=EnergyReal(p.tolerances.relativeEnergy);
    const EnergyReal relative=finite(suppliedRelative)&&suppliedRelative>=0?energyArithmetic::maximum(suppliedRelative,EnergyReal(32)*std::numeric_limits<EnergyReal>::epsilon()):suppliedRelative;
    const EnergyReal relativeFluxTolerance=relative==0?EnergyReal(0):
        (scale==std::numeric_limits<EnergyReal>::infinity()?scale:surfaceEnergyArithmetic::auditProduct(relative,scale));
    return surfaceEnergyArithmetic::auditAdd(surfaceEnergyArithmetic::auditQuotientProduct(p.tolerances.absoluteEnergy,
        in.area,(in.dt>0?EnergyReal(in.dt):EnergyReal(1))),relativeFluxTolerance);
}
// The owner copies immutable physics once per root solve. CUDA device atomics
// must never target an automatic/thread-local DeviceContext.
template<class Team,class=void> struct SurfaceProbeExecution {static constexpr int count=1,depth=1;};
template<class Team> struct SurfaceProbeExecution<Team,std::void_t<decltype(Team::probeCount)>> {static constexpr int count=Team::probeCount,depth=Team::probeDepth;};
template<class Evaluation,class=void> struct SurfaceBisectionProbeType {using type=ugkp::gpu::OrderedBisectionProbe<EnergyReal,1>;};
template<class Evaluation> struct SurfaceBisectionProbeType<Evaluation,std::void_t<typename Evaluation::BisectionProbes>> {using type=typename Evaluation::BisectionProbes;};
template<class Evaluation,class=void> struct HasSurfaceBisectionBatch:std::false_type {};
template<class Evaluation> struct HasSurfaceBisectionBatch<Evaluation,std::void_t<decltype(&Evaluation::evaluateBisection)>>:std::true_type {};
template<bool Track=false,class Evaluation=LegacySurfaceEvaluation,class Team=ugkwp::gaswall::detail::SerialWallTeam>
CHMT_HD inline bool bisectFeasibleSurfaceImpl(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const PhysicsConfig& trialPhysics,const MaterialPrimitive& material, EnergyReal filmT, EnergyReal lo, EnergyReal hi,
    SurfacePhysicsResult left, SurfacePhysicsResult right, SurfacePhysicsResult& out,SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr,
    const Team& team=Team{}) {
    if ((left.thermalResidual>0)==(right.thermalResidual>0)
        && left.thermalResidual!=0 && right.thermalResidual!=0) return false;
    if (left.thermalResidual==0) { out=left; return true; }
    if (right.thermalResidual==0) { out=right; return true; }
    typename SurfaceBisectionProbeType<Evaluation>::type probes;
    for (int iteration=0; iteration<p.numerics.surface.thermalMaximumIterations; ++iteration) {
        const EnergyReal T=EnergyReal(.5)*(lo+hi);
        SurfacePhysicsResult candidate;
        if(Track){trace->attempted=T;trace->hasAttempt=true;}
        bool valid=false;
        if constexpr(HasSurfaceBisectionBatch<Evaluation>::value)
            valid=Evaluation::evaluateBisection(in,trialPhysics,material,filmT,lo,hi,probes,candidate,numericalFailure);
        else valid=Evaluation::evaluate(in,trialPhysics,material,filmT,T,false,candidate,numericalFailure);
        if(!valid)return false;
        if(Track){trace->completedState=candidate;trace->hasCompleted=true;}
        const EnergyReal fluxTolerance=surfaceThermalFluxTolerance(in,p,candidate);
        const EnergyReal requestedTemperatureTolerance=surfaceEnergyArithmetic::auditTolerance(EnergyReal(p.tolerances.absoluteTemperature),EnergyReal(p.tolerances.relativeTemperature),T);
        const EnergyReal temperatureTolerance=finite(requestedTemperatureTolerance)&&requestedTemperatureTolerance>=0?energyArithmetic::maximum(requestedTemperatureTolerance,EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon()*energyArithmetic::absolute(T)):requestedTemperatureTolerance;
        if (energyArithmetic::absolute(candidate.thermalResidual)<=fluxTolerance
            && (hi-lo<=temperatureTolerance || candidate.thermalResidual==0)) {
            out=candidate; return true;
        }
        if (T==lo || T==hi || iteration+1==p.numerics.surface.thermalMaximumIterations) {
            if(team.leader())reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3804,
                candidate.thermalResidual,fluxTolerance);
            out=candidate;return true;
        }
        if ((candidate.thermalResidual>0)==(left.thermalResidual>0)) { lo=T; left=candidate; }
        else { hi=T; right=candidate; }
    }
    return false;
}
template<bool Track=false,class Evaluation=LegacySurfaceEvaluation>
CHMT_HD inline bool bisectFeasibleSurface(const SurfacePhysicsInput& in,const PhysicsConfig& p,
    const MaterialPrimitive& material,EnergyReal filmT,EnergyReal lo,EnergyReal hi,
    SurfacePhysicsResult left,SurfacePhysicsResult right,SurfacePhysicsResult& out,
    SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr) {
    if(!surfaceTemperatureConfiguration(in,p))return false;
    auto trialPhysics=p;
    trialPhysics.diagnostics=nullptr;trialPhysics.tolerances.diagnostics=nullptr;
    auto trialInput=in;trialInput.wallContext.failureStatus=nullptr;
    SurfacePhysicsResult candidate;
    if(!bisectFeasibleSurfaceImpl<Track,Evaluation>(trialInput,p,trialPhysics,material,filmT,lo,hi,left,right,candidate,trace,numericalFailure))return false;
    return Evaluation::evaluate(in,p,material,filmT,candidate.topTemperature,false,out,numericalFailure);
}
template<bool Track=false,class Evaluation=LegacySurfaceEvaluation,class Team=ugkwp::gaswall::detail::SerialWallTeam>
CHMT_HD inline bool findFeasibleSurfaceRootImpl(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const PhysicsConfig& trialPhysics,
    const MaterialPrimitive& material, EnergyReal filmT, EnergyReal lo, EnergyReal hi, SurfacePhysicsResult& out,SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr,
    const Team& team=Team{}) {
    if (!(hi>lo)) return false;
    const EnergyReal preferred=energyArithmetic::maximum(lo,energyArithmetic::minimum(hi,in.hasFilm && in.film.mass>0 ? filmT : material.temperature));
    bool configured=false;
    if(team.leader())configured=surfaceTemperatureConfiguration(in,p);
    if(!team.broadcast(configured))return false;
    auto trialInput=in;trialInput.wallContext.failureStatus=nullptr;
    SurfacePhysicsResult seed;
    if(Track){trace->attempted=preferred;trace->hasAttempt=true;}
    const bool seedValid=Evaluation::evaluate(trialInput,trialPhysics,material,filmT,preferred,false,seed,numericalFailure);
    if(Track&&seedValid){trace->completedState=seed;trace->hasCompleted=true;}
    // Only a roundoff-sized fraction of the ordinary heat-balance tolerance
    // qualifies for zero-update convergence. Do not relax the normal coupled
    // energy and temperature-bracket criteria for a finite physical imbalance.
    // Tiny nonzero reaction rates need not have a representable temperature
    // correction, and must not force a scan to a remote state.
    const EnergyReal seedTolerance=std::numeric_limits<EnergyReal>::epsilon()*surfaceThermalFluxTolerance(in,p,seed);
    if(seedValid&&(seed.thermalResidual==0||energyArithmetic::absolute(seed.thermalResidual)<=seedTolerance)){
        out=seed;return true;
    }
    SurfacePhysicsResult last=seed;bool hasValid=seedValid;
    // Invalid samples split feasible components; never bridge an observed hole.
    // Refine bracket discovery without changing the physical time interval.
    for (int subdivisions=p.numerics.surface.initialSubdivisions; subdivisions>0 && subdivisions<=p.numerics.surface.maximumSubdivisions;) {
        SurfacePhysicsResult previous;
        EnergyReal previousT=lo;
        bool previousValid=false;
        for (int i=0; i<=subdivisions; ++i) {
            const EnergyReal gridT=lo+(hi-lo)*static_cast<EnergyReal>(i)/subdivisions;
            const bool insert=i>0 && preferred>lo+(hi-lo)*(i-1)/subdivisions && preferred<gridT;
            for (int part=0; part<(insert ? 2 : 1); ++part) {
                const EnergyReal T=insert && part==0 ? preferred : gridT;
                SurfacePhysicsResult current;
                if(Track){trace->attempted=T;trace->hasAttempt=true;}
                const bool valid=Evaluation::evaluate(trialInput,trialPhysics,material,filmT,T,false,current,numericalFailure);
                            if(!valid){previousValid=false;continue;}
                last=current;hasValid=true;
                if(Track){trace->completedState=current;trace->hasCompleted=true;}
                if (current.thermalResidual==0) { out=current; return true; }
                if (valid && previousValid
                    && (current.thermalResidual>0)!=(previous.thermalResidual>0)) {
                    SurfacePhysicsResult root;
                    if(!bisectFeasibleSurfaceImpl<Track,Evaluation>(trialInput,p,trialPhysics,material,filmT,previousT,T,previous,current,root,trace,numericalFailure,team)){
                        if(team.broadcast(ugkp::diagnostics::hasFatal(p.diagnostics)))return false;
                        previousValid=false;continue;}
                    out=root;return true;
                }
                previousValid=valid;
                if (valid) { previous=current; previousT=T; }
            }
        }
        const long long nextSubdivisions=static_cast<long long>(subdivisions)*p.numerics.surface.subdivisionGrowth;
        // Exhaust the finite search budget without integer wrap or a nonadvancing schedule.
        if(nextSubdivisions<=subdivisions || nextSubdivisions>p.numerics.surface.maximumSubdivisions)break;
        subdivisions=static_cast<int>(nextSubdivisions);
    }
    if(!hasValid){
        // Retain a real failure record when no mathematical state is evaluable.
        SurfacePhysicsResult failed;
        if(!Evaluation::evaluate(in,team.leader()?p:trialPhysics,material,filmT,preferred,false,failed,numericalFailure)){
            if(team.leader()&&!ugkp::diagnostics::hasFatal(p.diagnostics))interfaceFailure(p,3860,preferred);
            return false;
        }
        return false;
    }
    // No bracket among feasible states is a finite residual budget,
    // so retain the final mathematical iterate, without selecting an earlier state.
    if(team.leader())reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3831,last.thermalResidual,0);
    out=last;return true;
}
// Scalar callers retain their original silent trial view. Cooperative owners
// supply one immutable device-owned copy to the implementation above, avoiding
// a large per-lane configuration copy at every face and root solve.
template<bool Track=false,class Evaluation=LegacySurfaceEvaluation,class Team=ugkwp::gaswall::detail::SerialWallTeam>
CHMT_HD inline bool findFeasibleSurfaceRoot(const SurfacePhysicsInput& in,const PhysicsConfig& p,
    const MaterialPrimitive& material,EnergyReal filmT,EnergyReal lo,EnergyReal hi,SurfacePhysicsResult& out,
    SurfaceRootTrace* trace=nullptr,bool* numericalFailure=nullptr,const Team& team=Team{}) {
    auto trialPhysics=p;
    trialPhysics.diagnostics=nullptr;trialPhysics.tolerances.diagnostics=nullptr;
    return findFeasibleSurfaceRootImpl<Track,Evaluation>(in,p,trialPhysics,material,filmT,
        lo,hi,out,trace,numericalFailure,team);
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
        || (in.useSweptGeometry && (!finite(in.gasSweptRate) || !finite(in.solidSweptRate))))) return surfaceGeometryFailure(p,in.area,in.gasDistance);
    const PhysicalReal normalResidual=energyArithmetic::maximum(energyArithmetic::absolute(mag(in.normal)-1),energyArithmetic::maximum(energyArithmetic::absolute(mag(in.gasNormal)-1),in.hasSolid?energyArithmetic::absolute(mag(in.solidNormal)-1):0));
    if(normalResidual>PhysicalReal(1e-12))surfaceNormalWarning(p,normalResidual);
    MaterialPrimitive material;
    if (in.hasSolid && !recoverMaterial(in.solid,in.solidVolume,p,material,nullptr,-1,numericalFailure)) return false;
    FilmAux filmAux=in.aux;
    const bool wet=in.hasFilm && in.film.mass>0;
    if (wet && !recoverFilm(in.film,in.area,in.aux.pressure,p,filmAux,nullptr,-1)) return false;
    if (!wet && !in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
    if (!wet && (!closeEnough(in.gasArea,in.solidArea,p.tolerances.absoluteGeometry,
        p.tolerances.relativeGeometry) || mag(in.gasNormal-in.solidNormal)>PhysicalReal(1e-10)))
        surfaceAreaWarning(p,in.gasArea-in.solidArea,mag(in.gasNormal-in.solidNormal));
    if (wet && p.material.enableSurfaceReactions) return interfaceFailure(p,3829,in.film.mass,-1,ugkp::diagnostics::Code::MissingOperator);
    // Pore collapse/bubbling into a film needs an additional mechanical model.
    // The first Stefan phase interface is a dense material phase boundary.
    if (in.hasSolid && p.material.enableMelting && Real(material.porosity)>p.tolerances.relativeGeometry) return interfaceFailure(p,3830,material.porosity,-1,ugkp::diagnostics::Code::MissingOperator);
    EnergyReal lo=0, hi=std::numeric_limits<EnergyReal>::max();
    for (int s=0; s<Ns; ++s) { lo=energyArithmetic::maximum(lo,p.species[s].Tmin); hi=energyArithmetic::minimum(hi,p.species[s].Tmax); }
    if (wet) { lo=energyArithmetic::maximum(lo,p.liquid.Tmin); hi=energyArithmetic::minimum(hi,p.liquid.Tmax); }
    if (in.hasSolid) for (int c=0; c<Nc; ++c) if (in.solid.condensed[c]>0) {
        lo=energyArithmetic::maximum(lo,p.condensed[c].Tmin); hi=energyArithmetic::minimum(hi,p.condensed[c].Tmax);
    }
    const EnergyReal filmT=wet ? filmAux.temperature : p.meltTemperature;
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
        || mag(in.gasNormal-in.normal)>PhysicalReal(1e-10)
        || (in.hasSolid && (!closeEnough(in.solidArea,in.area,0,p.tolerances.relativeGeometry)
            || mag(in.solidNormal-in.normal)>PhysicalReal(1e-10)));
    const Real rateTolerance=p.tolerances.absoluteGeometry/(in.dt>0 ? in.dt : 1);
    if (offsetGeometry && (Real(energyArithmetic::absolute(candidate.topSpeed)*in.gasArea)>rateTolerance
        || Real(energyArithmetic::absolute(candidate.solidSpeed)*in.solidArea)>rateTolerance
        || (in.useSweptGeometry && (std::abs(in.gasSweptRate)>rateTolerance
            || std::abs(in.solidSweptRate)>rateTolerance)))) {
        return failStatus(status,ErrorCode::Unsupported,-1,candidate.filmVolumeResidual,ErrorLocation::Face);
    }
    out=candidate; return true;
}
struct SurfacePacketIdentity {
    std::uint64_t step=0, geometry=0, face=0;
    int stage=0, gasCell=-1, solidCell=-1, filmFace=-1;
    EnergyReal oldFilmPV=0;
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
    const PhysicalReal dt=PhysicalReal(input.dt);
    const PhysicalReal scale=input.area*dt;
    const EnergyReal energyScale=EnergyReal(input.area)*EnergyReal(dt);
    const ExchangeKind primaryKind=result.wet ? ExchangeKind::GasFilm : ExchangeKind::GasSolid;
    ExchangePacket primary=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket pore=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket phase=surfacePacketIdentity(primaryKind,identity);
    for (int s=0; s<Ns; ++s) {
        primary.species[s]=scale*result.gasSpecies[s];
        primary.mass+=primary.species[s];
    }
    for (int c=0; c<Nc; ++c) primary.condensed[c]=scale*result.condensed[c];
    const EnergyReal mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?input.wallContext.mechanicalPressure:result.gasState.pressure;
    primary.momentum=result.gasState.velocity*primary.mass
        +(input.gasNormal*mechanicalPressure-result.traction)*(input.gasArea*dt);
    primary.advective=energyScale*result.gasAdvective;
    primary.conductive=energyScale*result.gasConductive;
    primary.pressureWork=energyScale*result.gasPressureWork;
    primary.viscousWork=energyScale*result.gasViscousWork;
    primary.energy=primary.advective+primary.conductive+primary.pressureWork+primary.viscousWork;
    primary.liquidKineticAdvection=result.wet ? energyScale*result.topKineticAdvection : 0;
    if (input.hasSolid) {
        pore=surfacePacketIdentity(ExchangeKind::PoreGas,identity);
        for (int s=0; s<Ns; ++s) {
            pore.species[s]=pore.pore[s]=scale*result.poreSpecies[s];
            pore.poreSweep[s]=scale*result.poreSweep[s];
            pore.mass+=pore.species[s];
        }
        pore.momentum=result.gasState.velocity*pore.mass;
        pore.energy=pore.advective=energyScale*result.poreEnergy;
    }
    if (input.hasSolid && result.wet) {
        phase=surfacePacketIdentity(ExchangeKind::SolidFilm,identity);
        phase.mass=scale*result.phaseMass;
        if (phase.mass!=0) phase.condensed[p.material.phaseCondensed]=phase.mass;
        for (int s=0; s<Ns; ++s) phase.species[s]=phase.mass*physicalFraction(p,p.material.phaseFilmY[s],s);
        phase.advective=energyScale*result.phaseAdvective;
        phase.conductive=energyScale*result.phaseConductive;
        phase.pressureWork=energyScale*result.phasePressureWork;
        phase.viscousWork=energyScale*result.phaseViscousWork;
        phase.energy=phase.advective+phase.conductive+phase.pressureWork+phase.viscousWork;
        phase.liquidKineticAdvection=energyScale*result.bottomKineticAdvection;

        // Exact event transfer: the disappearing film hands its actual remaining
        // internal energy to the solid. This is not a correction for R_K.
        FilmQ available=input.filmBase;
        available.mass+=dt*input.filmLocalRate.mass-primary.mass;
        for (int s=0; s<Ns; ++s) {
            available.species[s]+=dt*input.filmLocalRate.species[s]-primary.species[s];
        }
        const EnergyReal Uavailable=input.filmBase.enthalpy-identity.oldFilmPV
            +EnergyReal(dt)*input.filmLocalRate.enthalpy-primary.energy+primary.liquidKineticAdvection
            +energyScale*result.radiationFlux;
        if (phase.mass<0 && -phase.mass>=available.mass && available.mass>=0) {
            for (int s=0; s<Ns; ++s) {
                const PhysicalReal expected=available.mass*physicalFraction(p,p.material.phaseFilmY[s],s);
                const PhysicalReal residual=available.species[s]-expected;
                const PhysicalReal tolerance=p.tolerances.absoluteMass+p.tolerances.relativeMass
                    *energyArithmetic::maximum(energyArithmetic::absolute(available.species[s]),energyArithmetic::absolute(expected));
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
