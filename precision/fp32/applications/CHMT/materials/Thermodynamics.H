#ifndef CHMT_MATERIALS_THERMODYNAMICS_H
#define CHMT_MATERIALS_THERMODYNAMICS_H
#include "core/Configuration.H"
#include "core/AuditArithmetic.H"
#include "materials/GasThermoAdapter.H"
#include <string>
#include "../../../common/gasTransport/MolecularConductivity.H"
namespace chmt {
namespace energyArithmetic {
CHMT_HD inline EnergyReal absolute(EnergyReal x){return x<0?-x:x;}
CHMT_HD inline EnergyReal maximum(EnergyReal a,EnergyReal b){return a>b?a:b;}
CHMT_HD inline EnergyReal minimum(EnergyReal a,EnergyReal b){return a<b?a:b;}
CHMT_HD inline EnergyReal add(EnergyReal a,EnergyReal b){return a+b;}
CHMT_HD inline EnergyReal difference(EnergyReal a,EnergyReal b){return a-b;}
CHMT_HD inline EnergyReal product(EnergyReal a,EnergyReal b){return a*b;}
CHMT_HD inline EnergyReal tolerance(EnergyReal a,EnergyReal r,EnergyReal scale,EnergyReal ulps=EnergyReal(32)){
    const EnergyReal floor=ulps*std::numeric_limits<EnergyReal>::epsilon();
    if(std::isfinite(r)&&r>=EnergyReal(0)&&r<floor)r=floor;
    return a+r*scale;
}
}

CHMT_HD inline PhysicalReal gasMolecularConductivity(const PhysicsConfig& p,PhysicalReal T,const PhysicalReal* Y){
    if(p.gasTransportCoefficients.prandtlSpecies<0)return EnergyReal(p.gasConductivity);
    EnergyReal mass[Ns]{};for(int j=0;j<Ns;++j)mass[j]=EnergyReal(Y[j]);
    const EnergyReal temperature=EnergyReal(T);
    const GasThermoAdapter<Ns> thermo(p.species);
    const EnergyReal cp=ugkwp::mixtureSpecificCp(mass,temperature,thermo.view());
    // Configuration tables are external double inputs. Each coefficient is cast
    // by the templated molecular operator before physical arithmetic.
    const auto model=molecularTransportView(p);
    const EnergyReal mu=ugkwp::molecularViscosity<Ns>(model,EnergyReal(p.gasViscosity),temperature,mass);
    return ugkwp::conductivityFromViscosityCpPr(mu,cp,
        ugkwp::molecularPrandtl<Ns>(model,EnergyReal(1),mass));
}

CHMT_HD inline PhysicalReal physicalFraction(const PhysicsConfig& p,PhysicalReal value,int component=-1) {
    if(!clipPhysicsMassFraction(p,value,3832,component)){
        using namespace ugkp::diagnostics;
        if(p.diagnostics)recordFatal(p.diagnostics,makeRecord(Code::NonFinite,3833,38,0,EntityKind::None,-1,component,value,0,0,Quantity::MassFraction));
        return undefinedValue();
    }
    return value;
}
CHMT_HD inline EnergyReal condensedE(const CondensedThermo& s,EnergyReal T) { return EnergyReal(s.e0)+EnergyReal(s.cp0)*T+EnergyReal(.5)*EnergyReal(s.cp1)*T*T; }
CHMT_HD inline EnergyReal liquidH(const PhysicsConfig& p,EnergyReal T,EnergyReal pressure) {
    return condensedE(p.liquid,T)+(pressure-EnergyReal(p.liquidReferencePressure))/EnergyReal(p.liquid.rho);
}
CHMT_HD inline EnergyReal filmInternalEnergy(const FilmQ& q,EnergyReal pressure,const PhysicsConfig& p) {
    return q.enthalpy-pressure*EnergyReal(q.mass)/EnergyReal(p.liquid.rho);
}
CHMT_HD inline bool validSpecies(const SpeciesThermo& s) {
    const GasThermoAdapter<1> thermo(&s);
    return ugkwp::validateSpeciesThermoView(thermo.view())==ugkwp::ThermoStatus::Success;
}
CHMT_HD inline bool validCondensed(const CondensedThermo& s) {
    if(!finite(s.rho)||s.rho<=0||!finite(s.cp0)||!finite(s.cp1)||!finite(s.e0)||!finite(s.Tmin)||!finite(s.Tmax)||s.Tmin<=0||s.Tmax<=s.Tmin)return false;
    return true;
}
CHMT_HD inline bool closeEnough(PhysicalReal a,PhysicalReal b,PhysicalReal absolute,PhysicalReal relative) {
    if(!finite(a)||!finite(b)||!finite(absolute)||!finite(relative)||absolute<0||relative<0)return false;
    const PhysicalReal residual=energyArithmetic::difference(a,b),tolerance=energyArithmetic::tolerance(absolute,relative,maxValue(absValue(a),absValue(b)));
    return finite(residual)&&finite(tolerance)&&absValue(residual)<=tolerance;
}
CHMT_HD inline EnergyReal caloricIntegral(EnergyReal a0,EnergyReal a1,EnergyReal a2,EnergyReal T) { return a0+T*(a1+EnergyReal(.5)*a2*T); }
struct CondensedEnergyEvaluator {
    EnergyReal a0,a1,a2;
    CHMT_HD ugkwp::ThermoStatus operator()(EnergyReal T,EnergyReal& energy,EnergyReal& capacity) const {
        energy=caloricIntegral(a0,a1,a2,T);capacity=a1+a2*T;
        if(!finite(T)||!finite(energy)||!finite(capacity))return ugkwp::ThermoStatus::NonFiniteInput;
        if(T<=0)return ugkwp::ThermoStatus::TemperatureOutOfRange;
        if(capacity<=0)return ugkwp::ThermoStatus::NonPositiveHeatCapacity;
        return ugkwp::ThermoStatus::Success;
    }
};
CHMT_HD inline bool materialInversionFailure(ugkp::diagnostics::DeviceContext* diagnostics,unsigned site,
    ugkwp::ThermoStatus status,EnergyReal target,EnergyReal T) {
    using namespace ugkp::diagnostics;
    if(diagnostics)recordFatal(diagnostics,makeRecord(status==ugkwp::ThermoStatus::NonFiniteInput?Code::NonFinite:Code::InvalidPhysicalState,
        site,38,0,EntityKind::None,-1,static_cast<int>(status),target,T,0,Quantity::InternalEnergy));
    return false;
}
template<bool Terminal=false,class OutputReal=PhysicalReal>
CHMT_HD inline bool invertCaloric(EnergyReal a0,EnergyReal a1,EnergyReal a2,EnergyReal target,EnergyReal lo,EnergyReal hi,const Tolerances& tol,OutputReal& temperature) {
    using namespace ugkp::diagnostics;
    EnergyReal result=0;
    const EnergyReal delta=target-a0;
    // This caloric law is quadratic. Select its positive-capacity root directly;
    // metadata endpoints are neither trial states nor admissibility preflights.
    if(!finite(delta)||!finite(a1)||!finite(a2))
        return materialInversionFailure(tol.diagnostics,3838,ugkwp::ThermoStatus::NonFiniteInput,target,delta);
    if(delta==0&&a0!=0&&finite(lo)&&lo>0&&caloricIntegral(a0,a1,a2,lo)==target&&a1+a2*lo>0){
        result=lo;
        recordWarning(tol.diagnostics,makeRecord(Code::AccuracyResidual,3848,38,0,EntityKind::None,-1,-1,target,a0,result,Quantity::InternalEnergy));
    }else if(a2==0){
        if(a1<=0)return materialInversionFailure(tol.diagnostics,3850,ugkwp::ThermoStatus::NonPositiveHeatCapacity,target,a1);
        result=delta/a1;
    }else{
        // Form sqrt(2*a2*delta) by square roots before multiplication, then
        // use hypot/rescaled subtraction; neither cp^2 nor 2*delta is formed.
        const EnergyReal cross=std::sqrt(EnergyReal(2))*std::sqrt(energyArithmetic::absolute(a2))*std::sqrt(energyArithmetic::absolute(delta));
        EnergyReal root=0;
        if((a2<0)==(delta<0))root=std::hypot(a1,cross);
        else if(energyArithmetic::absolute(a1)>cross){
            const EnergyReal ratio=cross/energyArithmetic::absolute(a1);
            root=energyArithmetic::absolute(a1)*std::sqrt((1-ratio)*(1+ratio));
        }
        if(!finite(root)||root<=0)
            return materialInversionFailure(tol.diagnostics,3851,finite(root)?ugkwp::ThermoStatus::NonPositiveHeatCapacity:ugkwp::ThermoStatus::NonFiniteInput,target,root);
        if(a1>=0){const EnergyReal scale=energyArithmetic::maximum(a1,root);result=((delta/scale)/(a1/scale+root/scale))*2;}
        else result=root/a2-a1/a2;
    }
    EnergyReal value=0,capacity=0;
    const auto status=CondensedEnergyEvaluator{a0,a1,a2}(result,value,capacity);
    if(status!=ugkwp::ThermoStatus::Success)return materialInversionFailure(tol.diagnostics,3852,status,target,result);
    const EnergyReal residual=energyArithmetic::difference(value,target);
    const EnergyReal tolerance=energyArithmetic::tolerance(tol.absoluteEnergy,tol.relativeEnergy,energyArithmetic::maximum(energyArithmetic::absolute(target),energyArithmetic::absolute(value)));
    if(energyArithmetic::absolute(residual)>tolerance)recordWarning(tol.diagnostics,makeRecord(Code::AccuracyResidual,3854,38,0,EntityKind::None,-1,-1,residual,tolerance,result,Quantity::InternalEnergy));
    if(result<lo||result>hi)
        recordWarning(tol.diagnostics,makeRecord(Code::AccuracyResidual,3820,38,0,EntityKind::None,-1,-1,result,result<lo?lo:hi,0,Quantity::Temperature));
    temperature=result;return true;
}
CHMT_HD inline ugkwp::ThermoInversionControls<EnergyReal> gasThermoControls(const Tolerances& tol) {
    ugkwp::ThermoInversionControls<EnergyReal> controls;
    controls.absoluteEnergyTolerance=tol.absoluteEnergy;controls.relativeEnergyTolerance=tol.relativeEnergy;
    controls.absoluteTemperatureTolerance=tol.absoluteTemperature;controls.relativeTemperatureTolerance=tol.relativeTemperature;
    return ugkwp::castThermoControls<EnergyReal>(controls);
}
template<bool Terminal=false,class OutputReal=PhysicalReal>
CHMT_HD inline bool invertMaterialEnergy(const MaterialEnergyEvaluator& energy,EnergyReal target,
    EnergyReal lo,EnergyReal hi,const Tolerances& tol,OutputReal& temperature,ugkp::diagnostics::DeviceContext* diagnostics=nullptr,int maximumIterations=100) {
    using namespace ugkp::diagnostics;
    // A valid sign bracket is mathematical solver state, not a physical trial.
    // If metadata endpoints do not bracket the target, retain extrapolating
    // Newton and its existing range warning; metadata is not an admission gate.
    auto bracketEnergy=energy;bracketEnergy.thermo.diagnostics=nullptr;
    EnergyReal left=lo,right=hi,leftEnergy=0,rightEnergy=0,leftCapacity=0,rightCapacity=0;
    const bool bracket=finite(lo)&&finite(hi)&&hi>lo
        &&bracketEnergy(lo,leftEnergy,leftCapacity)==ugkwp::ThermoStatus::Success
        &&bracketEnergy(hi,rightEnergy,rightCapacity)==ugkwp::ThermoStatus::Success
        &&leftEnergy<=target&&target<=rightEnergy;
    EnergyReal candidate=bracket?left+(right-left)*EnergyReal(.5):lo,residual=0;
    const EnergyReal energyTolerance=energyArithmetic::tolerance(tol.absoluteEnergy,tol.relativeEnergy,energyArithmetic::absolute(target));
    for(int iteration=0;iteration<maximumIterations;++iteration){
        EnergyReal value=0,capacity=0;const auto status=energy(candidate,value,capacity);
        if(status!=ugkwp::ThermoStatus::Success)return materialInversionFailure(diagnostics,3839,status,target,candidate);
        residual=energyArithmetic::difference(value,target);
        const bool representableCorrection=finite(residual)&&!(capacity<1&&energyArithmetic::absolute(residual)>std::numeric_limits<EnergyReal>::max()*capacity);
        const EnergyReal correction=representableCorrection?residual/capacity:std::numeric_limits<EnergyReal>::infinity();
        const EnergyReal temperatureTolerance=energyArithmetic::tolerance(tol.absoluteTemperature,tol.relativeTemperature,energyArithmetic::absolute(candidate),EnergyReal(8));
        if(energyArithmetic::absolute(residual)<=energyTolerance&&energyArithmetic::absolute(correction)<=temperatureTolerance){
            const EnergyReal thermalScale=energyArithmetic::product(capacity,candidate);
            if(finite(thermalScale)&&thermalScale!=0&&energyArithmetic::add(value,thermalScale)==value)
                recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,3855,38,0,EntityKind::None,-1,-1,thermalScale,value,candidate,Quantity::InternalEnergy));
            break;
        }
        if(bracket){if(value<target)left=candidate;else right=candidate;}
        EnergyReal next=energyArithmetic::difference(candidate,correction);
        if(bracket&&(!finite(next)||next<=left||next>=right))next=left+(right-left)*EnergyReal(.5);
        if(!finite(next))return materialInversionFailure(diagnostics,3853,ugkwp::ThermoStatus::NonFiniteInput,target,next);
        if(iteration+1==maximumIterations||next==candidate){
            recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,3849,38,0,EntityKind::None,-1,-1,residual,energyTolerance,candidate,Quantity::InternalEnergy));break;
        }
        candidate=next;
    }
    if(candidate<lo||candidate>hi)
        recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,3821,38,0,EntityKind::None,-1,-1,candidate,candidate<lo?lo:hi,0,Quantity::Temperature));
    temperature=candidate;return true;
}
CHMT_HD inline bool recoverGas(const GasQ& q,Real volume,const PhysicsConfig& p,GasPrimitive& out,DeviceStatus* status,int index) {
    const auto physicalFailure=[&](PhysicalReal value,int component,unsigned quantity) {
        using namespace ugkp::diagnostics;
        recordFatal(p.diagnostics,makeRecord(finite(value)?Code::InvalidPhysicalState:Code::NonFinite,
            3858,38,0,EntityKind::Cell,index,component,value,0,0,quantity));
        return failStatus(status,finite(value)?ErrorCode::InvalidInput:ErrorCode::Nonfinite,index,value,ErrorLocation::Cell);
    };
    if(!finite(volume)||volume<=0)return physicalFailure(volume,-1,ugkp::diagnostics::Quantity::Unspecified);
    if(!finite(q.mass)||q.mass<=0)return physicalFailure(q.mass,-1,ugkp::diagnostics::Quantity::Unspecified);
    // Diagnose a true density overflow before executing the trapping division.
    if(volume<1&&q.mass>std::numeric_limits<PhysicalReal>::max()*PhysicalReal(volume))
        return physicalFailure(std::numeric_limits<PhysicalReal>::infinity(),-1,ugkp::diagnostics::Quantity::Density);
    const PhysicalReal momentum[]={q.momentum.x,q.momentum.y,q.momentum.z};
    for(int d=0;d<3;++d)if(!finite(momentum[d]))return physicalFailure(momentum[d],d,ugkp::diagnostics::Quantity::Unspecified);
    if(!finite(q.energy))return physicalFailure(q.energy,-1,ugkp::diagnostics::Quantity::InternalEnergy);
    PhysicalReal sum=0; EnergyReal R=0,lo=0,hi=std::numeric_limits<EnergyReal>::max();
    PhysicalReal masses[Ns]{};
    for(int s=0;s<Ns;++s){
        PhysicalReal Y=q.species[s]/q.mass;
        if(!finite(Y))return physicalFailure(Y,s,ugkp::diagnostics::Quantity::MassFraction);
        if(!clipPhysicsMassFraction(p,Y,3810,s))return physicalFailure(Y,s,ugkp::diagnostics::Quantity::MassFraction);
        const PhysicalReal m=q.mass*Y;masses[s]=m;
        sum=energyArithmetic::add(sum,m);if(m==0)continue;
        if(!validSpecies(p.species[s]))return failStatus(status,ErrorCode::PropertyRange,index,s,ErrorLocation::Cell);
        const auto& t=p.species[s];R+=EnergyReal(m)*EnergyReal(t.R);
        lo=energyArithmetic::maximum(lo,EnergyReal(t.Tmin));hi=energyArithmetic::minimum(hi,EnergyReal(t.Tmax));
    }
    if(!closeEnough(sum,q.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3811,energyArithmetic::difference(sum,q.mass),0);
    const EnergyReal mass=EnergyReal(q.mass),ux=EnergyReal(q.momentum.x)/mass,uy=EnergyReal(q.momentum.y)/mass,uz=EnergyReal(q.momentum.z)/mass;
    const PhysicalVec3 velocity={ux,uy,uz};const EnergyReal kinetic=EnergyReal(.5)*mass*(ux*ux+uy*uy+uz*uz);
    const GasThermoAdapter<Ns> thermo(p.species);
    const MaterialEnergyEvaluator energy{thermo.view(p.diagnostics),masses,0,0,0,p.species};
    EnergyReal T=0;if(!invertMaterialEnergy(energy,q.energy-kinetic,lo,hi,p.tolerances,T,p.diagnostics,p.numerics.thermoMaximumIterations))
        return failStatus(status,ErrorCode::PropertyRange,index,q.energy-kinetic,ErrorLocation::Cell);
    GasPrimitive candidate{};candidate.rho=q.mass/PhysicalReal(volume);candidate.velocity=velocity;candidate.temperature=T;
    EnergyReal internal=0,capacity=0;if(energy(EnergyReal(T),internal,capacity)!=ugkwp::ThermoStatus::Success)return failStatus(status,ErrorCode::PropertyRange,index,T,ErrorLocation::Cell);
    const EnergyReal Rmix=R/mass,cv=capacity/mass;
    candidate.pressure=EnergyReal(candidate.rho)*Rmix*EnergyReal(T);candidate.soundSpeed=std::sqrt((EnergyReal(1)+Rmix/cv)*Rmix*EnergyReal(T));
    for(int s=0;s<Ns;++s)candidate.Y[s]=masses[s]/q.mass;
    if(!finite(candidate.rho))return physicalFailure(candidate.rho,-1,ugkp::diagnostics::Quantity::Density);
    if(!finite(candidate.pressure))return physicalFailure(candidate.pressure,-1,ugkp::diagnostics::Quantity::Pressure);
    if(!finite(candidate.soundSpeed))return physicalFailure(candidate.soundSpeed,-1,ugkp::diagnostics::Quantity::Velocity);
    out=candidate;return true;
}
CHMT_HD inline bool recoverGas(const GasQ& q,Real volume,const PhysicsConfig& p,GasPrimitive& out) { return recoverGas(q,volume,p,out,nullptr,-1); }


CHMT_HD inline GasQ invalidGas() {
    GasQ q{};q.mass=q.energy=undefinedValue();q.momentum={undefinedValue(),undefinedValue(),undefinedValue()};
    for(int s=0;s<Ns;++s)q.species[s]=undefinedValue();
    return q;
}
CHMT_HD inline GasQ conservativeGas(const GasPrimitive& w,Real volume,const PhysicsConfig& p) {
    if(!finite(volume)||volume<=0||!finite(w.rho)||w.rho<=0||!finite(w.velocity)||!finite(w.temperature)||w.temperature<=0)return invalidGas();
    PhysicalReal sum=0;GasQ q{};q.mass=w.rho*PhysicalReal(volume);q.momentum=q.mass*w.velocity;
    const EnergyReal ux=EnergyReal(w.velocity.x),uy=EnergyReal(w.velocity.y),uz=EnergyReal(w.velocity.z);
    q.energy=EnergyReal(.5)*EnergyReal(q.mass)*(ux*ux+uy*uy+uz*uz);
    for(int s=0;s<Ns;++s){
        PhysicalReal fraction=w.Y[s];if(!clipPhysicsMassFraction(p,fraction,3813,s))return invalidGas();
        sum+=fraction;
        q.species[s]=q.mass*fraction;if(fraction==0)continue;
        if(!validSpecies(p.species[s]))return invalidGas();
        q.energy+=EnergyReal(q.species[s])*speciesE(p.species[s],EnergyReal(w.temperature));
    }
    if(!finite(q.mass)||!finite(q.energy))return invalidGas();
    if(!closeEnough(sum,1,p.tolerances.relativeMass,p.tolerances.relativeMass))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3814,sum-1,0);
    return q;
}




template<bool Terminal=false,class OutputReal=PhysicalReal>
CHMT_HD inline bool recoverSolid(const SolidQ& q,const PhysicsConfig& p,OutputReal& out,DeviceStatus* status,int index,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if((!finite(q.energy)||!finite(q.porosity)))return failStatus(status,ErrorCode::InvalidInput,index,q.porosity,ErrorLocation::Cell);
    PhysicalReal mass=0;EnergyReal a0=0,a1=0,a2=0,lo=0,hi=std::numeric_limits<EnergyReal>::max();
    for(int r=0;r<Nr;++r)if((!finite(q.progress[r])||q.progress[r]<0))return failStatus(status,ErrorCode::Inventory,index,q.progress[r],ErrorLocation::Cell);
    for(int c=0;c<Nc;++c){
        PhysicalReal m=q.condensed[c];if((!finite(m)||m<0))return failStatus(status,ErrorCode::Inventory,index,m,ErrorLocation::Cell);
        if(m==0)continue;
        if((!validCondensed(p.condensed[c])))return failStatus(status,ErrorCode::PropertyRange,index,c,ErrorLocation::Cell);
        const auto& t=p.condensed[c];mass+=m;a0+=EnergyReal(m)*EnergyReal(t.e0);a1+=EnergyReal(m)*EnergyReal(t.cp0);a2+=EnergyReal(m)*EnergyReal(t.cp1);lo=energyArithmetic::maximum(lo,EnergyReal(t.Tmin));hi=energyArithmetic::minimum(hi,EnergyReal(t.Tmax));
    }
    for(int s=0;s<Ns;++s){
        PhysicalReal m=q.pore[s];if((!finite(m)||m<0))return failStatus(status,ErrorCode::Inventory,index,m,ErrorLocation::Cell);
        if(m==0)continue;
        if((!validSpecies(p.species[s]))){if(numericalFailure)*numericalFailure=p.species[s].model==ugkwp::SpeciesThermoModel::LinearCp||p.species[s].model==ugkwp::SpeciesThermoModel::NASA7;return failStatus(status,ErrorCode::PropertyRange,index,s,ErrorLocation::Cell);}
        const auto& t=p.species[s];mass+=m;lo=energyArithmetic::maximum(lo,EnergyReal(t.Tmin));hi=energyArithmetic::minimum(hi,EnergyReal(t.Tmax));
    }
    const GasThermoAdapter<Ns> thermo(p.species);
    const MaterialEnergyEvaluator energy{thermo.view(p.diagnostics),q.pore,a0,a1,a2,p.species};
    EnergyReal T=0;if((mass<=0)||!invertMaterialEnergy<Terminal>(energy,q.energy,lo,hi,p.tolerances,T,p.diagnostics,p.numerics.thermoMaximumIterations)){
        if(numericalFailure)*numericalFailure=false;
        return failStatus(status,ErrorCode::PropertyRange,index,q.energy,ErrorLocation::Cell);
    }
    out=T;return true;
}
template<class OutputReal>
CHMT_HD inline bool recoverSolid(const SolidQ& q,const PhysicsConfig& p,OutputReal& out) { return recoverSolid(q,p,out,nullptr,-1); }
template<bool Terminal=false>
CHMT_HD inline bool recoverFilm(const FilmQ& q,Real area,PhysicalReal pressure,const PhysicsConfig& p,FilmAux& out,DeviceStatus* status,int index) {
    if((!finite(q.mass)||q.mass<0||!finite(q.enthalpy)||!finite(area)||area<=0||!finite(pressure)||!validCondensed(p.liquid)||!finite(p.liquidReferencePressure)))
        return failStatus(status,ErrorCode::InvalidInput,index,q.mass,ErrorLocation::Face);
    PhysicalReal sum=0;for(int s=0;s<Ns;++s){
        if(!finite(q.species[s]))return failStatus(status,ErrorCode::Nonfinite,index,q.species[s],ErrorLocation::Face);
        if(q.mass>0){PhysicalReal fraction=q.species[s]/q.mass;
            if(!clipPhysicsMassFraction(p,fraction,3816,s))return failStatus(status,ErrorCode::Nonfinite,index,fraction,ErrorLocation::Face);
            sum=energyArithmetic::add(sum,q.mass*fraction);
        }else sum=energyArithmetic::add(sum,q.species[s]);
    }
    if(!closeEnough(sum,q.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
        reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3812,energyArithmetic::difference(sum,q.mass),0);
    FilmAux candidate=out;candidate.area=area;candidate.pressure=pressure;
    if(q.mass==0){
        if((q.enthalpy!=0||sum!=0))return failStatus(status,ErrorCode::Inventory,index,q.enthalpy,ErrorLocation::Face);
        candidate.thickness=0;candidate.temperature=undefinedValue();
        candidate.meanVelocity=candidate.baseVelocity;candidate.topVelocity=candidate.baseVelocity;out=candidate;return true;
    }
    const EnergyReal m=EnergyReal(q.mass);
    const EnergyReal target=q.enthalpy-m*(EnergyReal(pressure)-EnergyReal(p.liquidReferencePressure))/EnergyReal(p.liquid.rho);
    PhysicalReal T=0;if(!invertCaloric<Terminal>(m*EnergyReal(p.liquid.e0),m*EnergyReal(p.liquid.cp0),m*EnergyReal(p.liquid.cp1),target,p.liquid.Tmin,p.liquid.Tmax,p.tolerances,T))
        return failStatus(status,ErrorCode::PropertyRange,index,target,ErrorLocation::Face);
    candidate.thickness=Real(q.mass)/(Real(p.liquid.rho)*area);candidate.temperature=T;
    if((!finite(candidate.thickness)))return failStatus(status,ErrorCode::Nonfinite,index,candidate.thickness,ErrorLocation::Face);
    out=candidate;return true;
}
CHMT_HD inline bool recoverFilm(const FilmQ& q,Real area,PhysicalReal pressure,const PhysicsConfig& p,FilmAux& out) { return recoverFilm(q,area,pressure,p,out,nullptr,-1); }

template<bool Terminal=false>
CHMT_HD inline PhysicalReal reactionRate(const Reaction& r,const PhysicalReal* condensedMass,PhysicalReal volume,PhysicalReal T) {
    if((!finite(volume)||volume<=0||!finite(T)||T<=0||!finite(r.A)||r.A<=0||!finite(r.temperaturePower)||!finite(r.activationEnergy)||r.activationEnergy<0))return undefinedValue();
    PhysicalReal rate=r.A*volume*std::pow(T,r.temperaturePower)*std::exp(-r.activationEnergy/(universalGasConstant*T));
    for(int c=0;c<Nc;++c){
        if((!finite(condensedMass[c])||condensedMass[c]<0||!finite(r.order[c])||r.order[c]<0))return undefinedValue();
        if(r.order[c]>0)rate*=std::pow(condensedMass[c]/volume,r.order[c]);
    }
    return finite(rate)&&rate>=0?rate:undefinedValue();
}
CHMT_HD inline EnergyReal reactionFormationEnergy(const Reaction& r,const PhysicsConfig& p) {
    EnergyReal value=0;for(int c=0;c<Nc;++c)value+=EnergyReal(r.condensedNu[c])*EnergyReal(p.condensed[c].e0);
    for(int s=0;s<Ns;++s)if(r.gasNu[s]!=0){
        // This legacy diagnostic is a zero-temperature LinearCp intercept;
        // NASA7 has branch-specific integration constants, not that intercept.
        if(p.species[s].model!=ugkwp::SpeciesThermoModel::LinearCp)return undefinedValue();
        value+=EnergyReal(r.gasNu[s])*EnergyReal(p.species[s].e0);
    }
    return value;
}
inline bool validateReaction(const Reaction& r,const PhysicsConfig& p,int index,std::string& error) {
    const std::string prefix="reaction "+std::to_string(index)+": ";
    if(!finite(r.A)||r.A<=0||!finite(r.temperaturePower)||!finite(r.activationEnergy)||r.activationEnergy<0){error=prefix+"invalid/missing Arrhenius parameters";return false;}
    if(p.nElements<=0||p.nElements>Ne){error=prefix+"missing/invalid active element count";return false;}
    if(!finite(p.tolerances.relativeMass)||p.tolerances.relativeMass<0){error=prefix+"invalid mass-balance tolerance";return false;}
    PhysicalReal sum=0,scale=0,element[Ne]{},elementScale[Ne]{};bool consumes=false,produces=false;
    const auto addCoefficient=[&](PhysicalReal nu,const PhysicalReal* composition)->bool {
        if(!finite(nu))return false;
        sum+=nu;scale+=absValue(nu);
        if(!finite(sum)||!finite(scale))return false;
        consumes=consumes||nu<0;produces=produces||nu>0;
        PhysicalReal defined=0;
        for(int e=0;e<Ne;++e){
            const PhysicalReal n=composition[e];if(!finite(n)||n<0)return false;
            if(e<p.nElements){defined+=n;if(!finite(defined))return false;}
            const PhysicalReal product=nu*n;if(!finite(product))return false;
            element[e]+=product;elementScale[e]+=absValue(product);
            if(!finite(element[e])||!finite(elementScale[e]))return false;
        }
        return nu==0||defined>0;
    };
    for(int c=0;c<Nc;++c){
        if(!finite(r.order[c])||r.order[c]<0||(r.order[c]>0&&r.condensedNu[c]>=0)){
            error=prefix+"invalid condensed reaction order";return false;
        }
        if(!addCoefficient(r.condensedNu[c],p.condensed[c].element)){
            error=prefix+"missing/invalid/nonfinite condensed mass or element arithmetic";return false;
        }
    }
    for(int s=0;s<Ns;++s)if(!addCoefficient(r.gasNu[s],p.species[s].element)){
        error=prefix+"missing/invalid/nonfinite gas mass or element arithmetic";return false;
    }
    const PhysicalReal massTolerance=64*std::numeric_limits<PhysicalReal>::epsilon()+p.tolerances.relativeMass;
    if(!finite(massTolerance)||!consumes||!produces||scale==0){error=prefix+"missing or invalid stoichiometric operator";return false;}
    if(absValue(sum)/scale>massTolerance)reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3834,sum/scale,massTolerance,0,index);
    // Element residuals are mol/mol extent: their threshold is dimensionless
    // roundoff, not the absolute kg inventory tolerance used by time updates.
    for(int e=0;e<Ne;++e){
        const PhysicalReal threshold=64*std::numeric_limits<PhysicalReal>::epsilon()*maxValue(1,elementScale[e]);
        if(!finite(threshold)){error=prefix+"nonfinite element threshold "+std::to_string(e);return false;}
        if(absValue(element[e])>threshold)reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3835,element[e],threshold,0,e);
    }
    error.clear();return true;
}

// Restrict one molar extent before rebuilding ALL linked mass changes. No
// component clipping and no independent heat source accompanies formation e0.
CHMT_HD inline bool limitReactionExtent(const Reaction& r,const SolidQ& q,PhysicalReal requested,PhysicalReal& accepted,DeviceStatus* status=nullptr,int index=-1) {
    if(!finite(requested)||requested<0)return failStatus(status,ErrorCode::InvalidInput,index,requested,ErrorLocation::Cell);
    PhysicalReal extent=requested;
    for(int c=0;c<Nc;++c){
        if(!finite(q.condensed[c])||q.condensed[c]<0||!finite(r.condensedNu[c]))return failStatus(status,ErrorCode::Inventory,index,q.condensed[c],ErrorLocation::Cell);
        if(r.condensedNu[c]<0)extent=minValue(extent,q.condensed[c]/(-r.condensedNu[c]));
    }
    for(int s=0;s<Ns;++s){
        if(!finite(q.pore[s])||q.pore[s]<0||!finite(r.gasNu[s]))return failStatus(status,ErrorCode::Inventory,index,q.pore[s],ErrorLocation::Cell);
        if(r.gasNu[s]<0)extent=minValue(extent,q.pore[s]/(-r.gasNu[s]));
    }
    if(!finite(extent))return failStatus(status,ErrorCode::Nonfinite,index,extent,ErrorLocation::Cell);
    accepted=extent;return true;
}
CHMT_HD inline bool applyReactionExtent(const Reaction& r,int reactionIndex,PhysicalReal extent,SolidQ& q,DeviceStatus* status=nullptr,int index=-1) {
    if(reactionIndex<0||reactionIndex>=Nr||!finite(q.energy)||!finite(q.porosity))
        return failStatus(status,ErrorCode::InvalidInput,index,reactionIndex,ErrorLocation::Cell);
    PhysicalReal bound=0;if(!limitReactionExtent(r,q,extent,bound,status,index))return false;
    if(extent>bound)return failStatus(status,ErrorCode::Inventory,index,extent,ErrorLocation::Cell);
    SolidQ candidate=q;
    for(int c=0;c<Nc;++c){candidate.condensed[c]+=extent*r.condensedNu[c];if(!finite(candidate.condensed[c])||candidate.condensed[c]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.condensed[c],ErrorLocation::Cell);}
    for(int s=0;s<Ns;++s){candidate.pore[s]+=extent*r.gasNu[s];if(!finite(candidate.pore[s])||candidate.pore[s]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.pore[s],ErrorLocation::Cell);}
    candidate.progress[reactionIndex]+=extent;
    if(!finite(candidate.progress[reactionIndex])||candidate.progress[reactionIndex]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.progress[reactionIndex],ErrorLocation::Cell);
    q=candidate;return true;
}
struct ValidationContext {
    bool gasThermo=true;
    unsigned condensedMask=(1u<<Nc)-1;
    bool liquid=true;
};
inline bool validatePhysicsConfig(const PhysicsConfig& p,const ValidationContext& context,std::string& error) {
    for(int s=0;s<Ns;++s)if(context.gasThermo&&!validSpecies(p.species[s])){error="configuration gas species "+std::to_string(s)+": missing/invalid thermo table";return false;}
    for(int c=0;c<Nc;++c)if((context.condensedMask&(1u<<c))&&(!validCondensed(p.condensed[c])||!finite(p.condensed[c].conductivity)||p.condensed[c].conductivity<0)){error="configuration condensed constituent "+std::to_string(c)+": missing/invalid property table";return false;}
    if(p.nReactions<0||p.nReactions>Nr||p.nElements<0||p.nElements>Ne){error="configuration reaction/element count exceeds compiled first-model limits";return false;}
    if(p.enableReactions&&p.nReactions==0){error="configuration reactions enabled without any mechanism";return false;}
    for(int r=0;r<p.nReactions;++r)if(!validateReaction(p.reactions[r],p,r,error))return false;
    if(context.liquid&&p.enableFilm&&(!validCondensed(p.liquid)||!finite(p.liquid.conductivity)||p.liquid.conductivity<0
        ||!finite(p.liquidViscosity)||p.liquidViscosity<=0||!finite(p.liquidReferencePressure)
        ||((p.material.enableMelting||p.filmThermalMode==FilmThermalMode::ResolvedNormal)
            &&(!finite(p.meltTemperature)||p.meltTemperature<=0)))){
        error="configuration film: missing/invalid liquid thermo, viscosity, reference pressure or melt temperature";return false;
    }
    if(!finite(p.permeability)||p.permeability<0||!finite(p.poreViscosity)||p.poreViscosity<0||(p.permeability>0&&p.poreViscosity<=0)
        ||!finite(p.gasViscosity)||p.gasViscosity<0||!finite(p.gasConductivity)||p.gasConductivity<0||!finite(p.gravity)){
        error="configuration transport/gravity: invalid coefficient";return false;
    }
    for(int s=0;s<Ns;++s)if(!finite(p.gasDiffusivity[s])||p.gasDiffusivity[s]<0){error="configuration diffusivity species "+std::to_string(s)+": invalid coefficient";return false;}
    if(!finite(p.emissivity)||!finite(p.ambientTemperature)||p.ambientTemperature<0||(p.enableRadiation&&p.ambientTemperature<=0)){
        error="configuration radiation: invalid emissivity or missing ambient temperature";return false;
    }
    const Real absolute[]={p.tolerances.absoluteMass,p.tolerances.absoluteEnergy,p.tolerances.absoluteTemperature,p.tolerances.absoluteGeometry};
    const Real relative[]={p.tolerances.relativeMass,p.tolerances.relativeEnergy,p.tolerances.relativeTemperature,p.tolerances.relativeGeometry};
    for(int i=0;i<4;++i)if(!finite(absolute[i])||absolute[i]<0||!finite(relative[i])||relative[i]<0||absolute[i]+relative[i]==0){error="configuration tolerance "+std::to_string(i)+": must be finite, nonnegative and nonzero";return false;}
    if(!finite(p.minDt)||p.minDt<=0||!finite(p.maxDt)||p.maxDt<p.minDt||!finite(p.cfl)||p.cfl<=0
        ||(p.spatialOrder!=1&&p.spatialOrder!=2)){error="configuration timestep/order: required explicit positive minDt, maxDt, CFL and supported order";return false;}
    if(p.enableSst){
        const auto& s=p.sst;
        const PhysicalReal values[]={s.a1,s.betaStar,s.beta1,s.beta2,s.gamma1,s.gamma2,s.sigmaK1,s.sigmaK2,s.sigmaOmega1,s.sigmaOmega2,s.turbulentPrandtl,s.turbulentSchmidt,s.productionLimit};
        for(PhysicalReal value:values)if(!finite(value)||value<=0){error="configuration SST: invalid model constant";return false;}
        if(!finite(s.minimumK)||s.minimumK<0||!finite(s.minimumOmega)||s.minimumOmega<0){error="configuration SST: invalid admissibility threshold";return false;}
    }
    if(p.reconstruction!=ReconstructionMode::LimitedLinear&&p.reconstruction!=ReconstructionMode::SmoothVerification){error="configuration reconstruction: unsupported policy";return false;}
    if(p.filmThermalMode!=FilmThermalMode::ThicknessAveraged&&p.filmThermalMode!=FilmThermalMode::ResolvedNormal){error="configuration film: unsupported thermal mode";return false;}
    if(p.meshMotion.policy!=MeshMotionPolicy::Static&&p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession&&p.meshMotion.policy!=MeshMotionPolicy::PrescribedSinusoidal){error="configuration mesh motion: unsupported policy";return false;}
    if(!finite(p.meshMotion.amplitude)||!finite(p.meshMotion.spatialWaveNumber)||!finite(p.meshMotion.spatialOrigin)||!finite(p.meshMotion.angularFrequency)||!finite(p.meshMotion.timeOrigin)){error="configuration mesh motion: nonfinite input";return false;}
    if(!p.enableGas&&(p.enableSst||p.enableParticles)){error="SST/particles require external gas enabled";return false;}
    error.clear();return true;
}
inline bool validatePhysicsConfig(const PhysicsConfig& p,std::string& error) {
    return validatePhysicsConfig(p,ValidationContext{},error);
}
} // namespace chmt
#endif
