#ifndef CHMT_MATERIALS_MATERIALREACTIONMATH_H
#define CHMT_MATERIALS_MATERIALREACTIONMATH_H
#include "materials/DryMaterialAdapter.H"
namespace chmt {
enum class MaterialReactionCode : int {
    Success=0,InvalidControls,BaseRecovery,LocalRecovery,CounterOverflow,NonadvancingStep
};
struct MaterialReactionStatus {
    MaterialReactionCode code=MaterialReactionCode::Success;
    bool numericalFailure=true;
    std::uint64_t acceptedSteps=0;
};
inline const char* materialReactionMessage(MaterialReactionCode code) {
    switch(code){
        case MaterialReactionCode::Success:return "";
        case MaterialReactionCode::InvalidControls:return "invalid reaction local control";
        case MaterialReactionCode::BaseRecovery:return "reaction base EOS invalid";
        case MaterialReactionCode::LocalRecovery:return "reaction local EOS/admissibility failure";
        case MaterialReactionCode::CounterOverflow:return "reaction local substep counter overflow";
        case MaterialReactionCode::NonadvancingStep:return "reaction local timestep cannot advance time";
    }
    return "reaction local EOS/admissibility failure";
}
// One cell's independent physical clock, using the original dry Euler map.
// The warning budget never truncates integration. A failed operation leaves
// the caller's candidate unchanged; this is not a retry or alternate method.
CHMT_HD inline MaterialReactionStatus advanceMaterialReactionsMath(const SolidQ& base,
    Real volume,const PhysicsConfig& p,Real interval,Real maxStep,Real cfl,
    int warningBudget,SolidQ& output) {
    MaterialReactionStatus status;
    if(!(interval>0)||!finite(interval)||maxStep<0||!finite(maxStep)
        ||!(finite(cfl)&&cfl>0)||warningBudget<=0){
        status.code=MaterialReactionCode::InvalidControls;return status;
    }
    SolidQ state=base;MaterialPrimitive material;
    if(!recoverMaterial<false>(state,volume,p,material,nullptr,-1,&status.numericalFailure)){
        status.code=MaterialReactionCode::BaseRecovery;return status;
    }
    if(!p.enableReactions){output=base;return status;}
    // Preserve count/domain diagnostic precedence before reading the network.
    ReactionStepControl initialControl;
    if(!reactionStepControl(state,volume,p,cfl,initialControl,&status.numericalFailure,
        dryAdapter::CoreReactionRates{})){
        status.code=MaterialReactionCode::LocalRecovery;return status;
    }
    dry::Reaction<Nc,Ns> reactions[Nr]{};
    for(int r=0;r<p.nReactions;++r)reactions[r]=dryAdapter::reaction(p.reactions[r]);
    const dry::ReactionView<Nc,Ns> network{reactions,static_cast<std::size_t>(p.nReactions)};
    Real elapsed=0;std::uint64_t count=0;
    while(elapsed<interval){
        if(count==static_cast<std::uint64_t>(warningBudget))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,30900,count,warningBudget,count);
        if(count==std::numeric_limits<std::uint64_t>::max()){
            status.code=MaterialReactionCode::CounterOverflow;return status;
        }
        const Real maximumStep=minValue(interval-elapsed,maxStep>0?maxStep:interval);
        SolidQ trial;Real dt=0;bool active=false;
        if(!dryAdapter::reactionStep(state,volume,p,network,maximumStep,cfl,trial,dt,
            active,&status.numericalFailure)){
            status.code=MaterialReactionCode::LocalRecovery;return status;
        }
        if(!active){output=state;status.acceptedSteps=count;return status;}
        if(!finite(dt)||!(dt>0)||!finite(elapsed+dt)||elapsed+dt==elapsed){
            status.code=MaterialReactionCode::NonadvancingStep;return status;
        }
        state=trial;elapsed+=dt;++count;
        if(interval-elapsed<=8*std::numeric_limits<Real>::epsilon()*interval)elapsed=interval;
    }
    output=state;status.acceptedSteps=count;return status;
}
}
#endif
