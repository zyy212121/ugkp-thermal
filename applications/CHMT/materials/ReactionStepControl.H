#ifndef CHMT_MATERIALS_REACTIONSTEPCONTROL_H
#define CHMT_MATERIALS_REACTIONSTEPCONTROL_H
#include "materials/Reaction.H"
namespace chmt {
// Shared host/device timestep mathematics. No state is committed by these
// helpers: CPU chemistry and the retained CUDA explicit controller use the same
// constant-U rate bound and the same conservative Euler defect estimator.
struct ReactionStepControl {
    Real temperature=0,capacity=0,massTolerance=0,dtLimit=std::numeric_limits<Real>::max();
    Real sensitivity=0,extentEnergy[Nr]{},stoichiometricMass[Nr]{};
    bool active=false;
};
// A first-order factor has a finite derivative at zero. Singular fractional
// factors have no finite Jacobian bound; the trial-step defect controls them.
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
    Real cfl,ReactionStepControl& output) {
    if(!(cfl>0&&cfl<=1)||p.nReactions<0||p.nReactions>Nr)return false;
    MaterialPrimitive material;if(!recoverMaterial(state,volume,p,material))return false;
    ReactionStepControl c;c.temperature=material.temperature;
    Real mass=0,a1=0,a2=0;
    for(int k=0;k<Nc;++k){mass+=state.condensed[k];a1+=state.condensed[k]*p.condensed[k].cp0;a2+=state.condensed[k]*p.condensed[k].cp1;}
    for(int s=0;s<Ns;++s){mass+=state.pore[s];if(state.pore[s]>0)a1+=state.pore[s]*speciesCv(p.species[s],c.temperature);}
    c.capacity=a1+a2*c.temperature;c.massTolerance=maxValue(p.tolerances.absoluteMass,64*std::numeric_limits<Real>::epsilon()*mass);
    if(!(c.capacity>0)||!finite(c.capacity)||!finite(c.massTolerance)||c.massTolerance<0)return false;
    if(!p.enableReactions){output=c;return true;}
    SolidQ loss;Real stageRates[Nr]{};bool negligible[Nr]{};
    for(int r=0;r<p.nReactions;++r){const auto& reaction=p.reactions[r];const Real rate=reactionRate(reaction,state.condensed,volume,c.temperature);
        if(!finite(rate)||rate<0)return false;
        stageRates[r]=rate;c.active=c.active||rate>0;
        Real donorExtent=std::numeric_limits<Real>::max();bool fractionalDonor=false;
        for(int k=0;k<Nc;++k){const Real nu=reaction.condensedNu[k];loss.condensed[k]+=rate*maxValue(0,-nu);c.extentEnergy[r]+=nu*condensedE(p.condensed[k],c.temperature);
            c.stoichiometricMass[r]+=absValue(nu);if(nu<0)donorExtent=minValue(donorExtent,state.condensed[k]/(-nu));
            fractionalDonor=fractionalDonor||(nu<0&&reaction.order[k]>0&&reaction.order[k]<1&&state.condensed[k]<=c.massTolerance);}
        for(int s=0;s<Ns;++s){const Real nu=reaction.gasNu[s];loss.pore[s]+=rate*maxValue(0,-nu);c.extentEnergy[r]+=nu*speciesE(p.species[s],c.temperature);
            c.stoichiometricMass[r]+=absValue(nu);if(nu<0)donorExtent=minValue(donorExtent,state.pore[s]/(-nu));}
        // Only tolerance-sized fractional donor changes may bypass a singular
        // bound at extinction. The aggregate limiter and defect still apply;
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
        if(!finite(row))return false;
        c.sensitivity=maxValue(c.sensitivity,row);
    }
    for(int k=0;k<Nc;++k)if(loss.condensed[k]>0&&state.condensed[k]>c.massTolerance)c.dtLimit=minValue(c.dtLimit,cfl*state.condensed[k]/loss.condensed[k]);
    for(int s=0;s<Ns;++s)if(loss.pore[s]>0&&state.pore[s]>c.massTolerance)c.dtLimit=minValue(c.dtLimit,cfl*state.pore[s]/loss.pore[s]);
    if(c.sensitivity>0)c.dtLimit=minValue(c.dtLimit,cfl/c.sensitivity);
    if(!(c.dtLimit>0)||!finite(c.dtLimit))return false;
    output=c;return true;
}
// Select one admissible local step. The full Euler state is returned; the two
// half steps are only an error estimate, not a change to the accepted method.
// Failure leaves output, step and active unchanged. The bounded retry loop is
// shared with CUDA timestep selection, not a second GPU chemistry integrator.
CHMT_HD inline bool reactionStepCandidate(const SolidQ& state,Real volume,const PhysicsConfig& p,
    Real maximumStep,Real cfl,SolidQ& output,Real& step,bool& active) {
    if(!(maximumStep>0)||!finite(maximumStep))return false;
    ReactionStepControl c;if(!reactionStepControl(state,volume,p,cfl,c))return false;
    if(!c.active){output=state;step=maximumStep;active=false;return true;}
    Real dt=minValue(maximumStep,c.dtLimit);
    for(int retry=0;retry<40;++retry){if(!finite(dt)||!(dt>0)||dt/2==0)break;
        SolidQ full,half,refined;Real rates[Nr]{},unused[Nr]{};MaterialPrimitive fullState,refinedState;
        if(advanceLocalReactions(state,volume,p,dt,full,rates)
            &&advanceLocalReactions(state,volume,p,dt/2,half,unused)
            &&advanceLocalReactions(half,volume,p,dt/2,refined,unused)
            &&recoverMaterial(full,volume,p,fullState)&&recoverMaterial(refined,volume,p,refinedState)){
            Real defect=0,variation=0,thermalVariation=0;
            for(int k=0;k<Nc;++k)defect+=absValue(full.condensed[k]-refined.condensed[k]);
            for(int s=0;s<Ns;++s)defect+=absValue(full.pore[s]-refined.pore[s]);
            for(int r=0;r<p.nReactions;++r){variation+=dt*rates[r]*c.stoichiometricMass[r];thermalVariation+=dt*rates[r]*absValue(c.extentEnergy[r])/c.capacity;}
            const Real temperatureTolerance=p.tolerances.absoluteTemperature+p.tolerances.relativeTemperature
                *maxValue(absValue(fullState.temperature),absValue(refinedState.temperature));
            if(defect<=Real(.25)*cfl*variation+c.massTolerance
                &&absValue(fullState.temperature-refinedState.temperature)<=Real(.25)*cfl*thermalVariation+temperatureTolerance){
                output=full;step=dt;active=true;return true;}
        }
        dt*=.5;
    }
    return false;
}
}
#endif
