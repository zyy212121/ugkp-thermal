#ifndef CHMT_MATERIALS_REACTIONSTEPCONTROL_H
#define CHMT_MATERIALS_REACTIONSTEPCONTROL_H
#include "materials/Reaction.H"
namespace chmt {
CHMT_HD inline bool reactionControlFailure(const PhysicsConfig& p,ugkp::diagnostics::Code code,
    unsigned site,Real value,int component=-1,unsigned quantity=ugkp::diagnostics::Quantity::Unspecified) {
    using namespace ugkp::diagnostics;
    recordFatal(p.diagnostics,makeRecord(code,site,4,0,EntityKind::Cell,-1,component,value,0,0,quantity));
    return false;
}
// Shared host/device timestep mathematics. No state is committed by these
// helpers: CPU chemistry and the retained CUDA explicit controller use the same
// constant-U rate bound and conservative Euler advancement.
struct ReactionStepControl {
    Real temperature=0,capacity=0,massTolerance=0,dtLimit=std::numeric_limits<Real>::max();
    Real sensitivity=0,extentEnergy[Nr]{},stoichiometricMass[Nr]{};
    bool active=false;
};
// A first-order factor has a finite derivative at zero. Singular fractional
// factors have no finite Jacobian bound; inventory limiting still applies.
CHMT_HD inline Real reactionMassDerivative(const Reaction& reaction,int component,const SolidQ& q,
    Real volume,Real temperature,Real rate) {
    const Real order=reaction.order[component],mass=q.condensed[component];
    if(order==0)return 0;
    if(mass>0)return rate*order/mass;
    if(order!=1)return 0;
    Real derivative=reaction.A*::pow(temperature,reaction.temperaturePower)
        *::exp(-reaction.activationEnergy/(universalGasConstant*temperature));
    for(int k=0;k<Nc;++k)if(k!=component&&reaction.order[k]>0)
        derivative*=::pow(q.condensed[k]/volume,reaction.order[k]);
    return derivative;
}
CHMT_HD inline bool reactionStepControl(const SolidQ& state,Real volume,const PhysicsConfig& p,
    Real cfl,ReactionStepControl& output,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if(!(finite(cfl)&&cfl>0))return reactionControlFailure(p,finite(cfl)?ugkp::diagnostics::Code::InvalidPhysicalState:ugkp::diagnostics::Code::NonFinite,4110,cfl);
    if(p.nReactions<0||p.nReactions>Nr)return reactionControlFailure(p,ugkp::diagnostics::Code::InvalidIndex,4111,p.nReactions);
    MaterialPrimitive material;if(!recoverMaterial<false>(state,volume,p,material,nullptr,-1,numericalFailure))return false;
    ReactionStepControl c;c.temperature=material.temperature;
    Real mass=0,a1=0,a2=0;
    for(int k=0;k<Nc;++k){mass+=state.condensed[k];a1+=state.condensed[k]*p.condensed[k].cp0;a2+=state.condensed[k]*p.condensed[k].cp1;}
    for(int s=0;s<Ns;++s){mass+=state.pore[s];if(state.pore[s]>0)a1+=state.pore[s]*(speciesCv(p.species[s],c.temperature));}
    c.capacity=a1+a2*c.temperature;c.massTolerance=maxValue(p.tolerances.absoluteMass,64*std::numeric_limits<Real>::epsilon()*mass);
    if(!(c.capacity>0)||!finite(c.capacity))return reactionControlFailure(p,!finite(c.capacity)?ugkp::diagnostics::Code::NonFinite:c.capacity==0?ugkp::diagnostics::Code::ZeroDenominator:ugkp::diagnostics::Code::InvalidPhysicalState,4112,c.capacity);
    if(!finite(c.massTolerance)||c.massTolerance<0)return reactionControlFailure(p,!finite(c.massTolerance)?ugkp::diagnostics::Code::NonFinite:ugkp::diagnostics::Code::InvalidPhysicalState,4113,c.massTolerance);
    if(!p.enableReactions){output=c;return true;}
    SolidQ loss;Real stageRates[Nr]{};bool negligible[Nr]{};
    for(int r=0;r<p.nReactions;++r){const auto& reaction=p.reactions[r];const Real rate=reactionRate<false>(reaction,state.condensed,volume,c.temperature);
        if(!finite(rate)||rate<0)return reactionControlFailure(p,!finite(rate)?ugkp::diagnostics::Code::NonFinite:ugkp::diagnostics::Code::InvalidPhysicalState,4114,rate,r);
        stageRates[r]=rate;c.active=c.active||rate>0;
        Real donorExtent=std::numeric_limits<Real>::max();bool fractionalDonor=false;
        for(int k=0;k<Nc;++k){const Real nu=reaction.condensedNu[k];loss.condensed[k]+=rate*maxValue(0,-nu);c.extentEnergy[r]+=nu*condensedE(p.condensed[k],c.temperature);
            c.stoichiometricMass[r]+=absValue(nu);if(nu<0)donorExtent=minValue(donorExtent,state.condensed[k]/(-nu));
            fractionalDonor=fractionalDonor||(nu<0&&reaction.order[k]>0&&reaction.order[k]<1&&state.condensed[k]<=c.massTolerance);}
        for(int s=0;s<Ns;++s){const Real nu=reaction.gasNu[s];loss.pore[s]+=rate*maxValue(0,-nu);c.extentEnergy[r]+=nu*speciesE(p.species[s],c.temperature);
            c.stoichiometricMass[r]+=absValue(nu);if(nu<0)donorExtent=minValue(donorExtent,state.pore[s]/(-nu));}
        // Only tolerance-sized fractional donor changes may bypass a singular
        // bound at extinction. The aggregate limiter still applies;
        // no component inventory is clipped, discarded, or artificially seeded.
        negligible[r]=fractionalDonor&&rate>0&&donorExtent*c.stoichiometricMass[r]<=c.massTolerance;
    }
    if(!c.active){output=c;return true;}
    // Infinity norm of d(rate_r)/d(extent_j): cross-reaction composition terms
    // and temperature response dT/dextent_j=-sum(nu_j*e)/capacity at fixed U.
    for(int r=0;r<p.nReactions;++r)if(!negligible[r]){
        const auto& reaction=p.reactions[r];Real row=0,dm[Nc]{};
        for(int k=0;k<Nc;++k)dm[k]=reactionMassDerivative(reaction,k,state,volume,c.temperature,stageRates[r]);
        const Real dT=stageRates[r]*(reaction.temperaturePower/c.temperature
            +reaction.activationEnergy/(universalGasConstant*c.temperature*c.temperature));
        for(int j=0;j<p.nReactions;++j){Real entry=-dT*c.extentEnergy[j]/c.capacity;
            for(int k=0;k<Nc;++k)entry+=dm[k]*p.reactions[j].condensedNu[k];
            row+=absValue(entry);}
        if(!finite(row))return reactionControlFailure(p,ugkp::diagnostics::Code::NonFinite,4115,row,r);
        c.sensitivity=maxValue(c.sensitivity,row);
    }
    for(int k=0;k<Nc;++k)if(loss.condensed[k]>0&&state.condensed[k]>c.massTolerance)c.dtLimit=minValue(c.dtLimit,cfl*state.condensed[k]/loss.condensed[k]);
    for(int s=0;s<Ns;++s)if(loss.pore[s]>0&&state.pore[s]>c.massTolerance)c.dtLimit=minValue(c.dtLimit,cfl*state.pore[s]/loss.pore[s]);
    if(c.sensitivity>0)c.dtLimit=minValue(c.dtLimit,cfl/c.sensitivity);
    if(!(c.dtLimit>0)||!finite(c.dtLimit))return reactionControlFailure(p,!finite(c.dtLimit)?ugkp::diagnostics::Code::NonFinite:ugkp::diagnostics::Code::InvalidPhysicalState,4116,c.dtLimit);
    output=c;return true;
}
// Advance one full conservative Euler step selected from the rate/CFL bound.
// There is no defect-estimator integration or failed-step repetition.
CHMT_HD inline bool reactionStepCandidate(const SolidQ& state,Real volume,const PhysicsConfig& p,
    Real maximumStep,Real cfl,SolidQ& output,Real& step,bool& active,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if(!(maximumStep>0)||!finite(maximumStep))return reactionControlFailure(p,!finite(maximumStep)?ugkp::diagnostics::Code::NonFinite:ugkp::diagnostics::Code::InvalidPhysicalState,4117,maximumStep);
    ReactionStepControl c;if(!reactionStepControl(state,volume,p,cfl,c,numericalFailure))return false;
    if(!c.active){output=state;step=maximumStep;active=false;return true;}
    const Real dt=minValue(maximumStep,c.dtLimit);
    SolidQ full;Real rates[Nr]{};
    if(!advanceLocalReactions(state,volume,p,dt,full,rates,nullptr,-1,numericalFailure))return false;
    MaterialPrimitive endpoint;
    if(!recoverMaterial(full,volume,p,endpoint,nullptr,-1,numericalFailure))return false;
    output=full;step=dt;active=true;return true;
}
}
#endif
