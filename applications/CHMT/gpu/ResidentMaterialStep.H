#pragma once
#include "gpu/ResidentWallProgram.H"
#include "../../../common/IntervalSlice.H"
#include "../../../common/gasTransport/GasDriveSelection.H"
namespace chmt {
enum class MaterialStepSelectionError : int {None,Drive,Transport,NonAdvancing};
struct ResidentMaterialStep {
    ugkp::CompensatedTime begin{},end{};
    Real dt=0;
    std::size_t drive=0;
    bool active=false,finalStep=false;
    MaterialStepSelectionError error=MaterialStepSelectionError::None;
};
// Existing material policy, separated from the host loop. The gas-history
// selector and physical transport limit supply device results between phases.
CHMT_HD inline ResidentMaterialStep requestMaterialStep(ugkp::CompensatedTime begin,
    ugkp::CompensatedTime total,Real maxSubstep,Real tolerance,Real tractionScale,
    ugkwp::GasDriveControls& drive){
    ResidentMaterialStep step;step.begin=step.end=begin;step.active=begin<total;
    if(!step.active)return step;
    const Real remaining=ugkp::boundedTimeDuration(total,begin);
    const Real maximum=maxSubstep>0?maxSubstep:total.value();
    step.dt=remaining<maximum?remaining:maximum;
    drive={begin,begin+step.dt,total,tolerance,tractionScale};return step;
}
CHMT_HD inline void alignMaterialStep(ResidentMaterialStep& step,const ugkwp::GasDriveSelection& drive,
    const ResidentWallKnotTime* knots,std::size_t count){
    if(!step.active||step.error!=MaterialStepSelectionError::None)return;
    if(drive.error!=ugkwp::GasDriveSelectionError::None){step.error=MaterialStepSelectionError::Drive;return;}
    step.drive=drive.held;step.dt=ugkp::boundedTimeDuration(drive.end,step.begin);
    // Retain original knot order and strict interior comparison, including
    // unrepresentable intermediate endpoints; do not add a microstep cap.
    for(std::size_t i=0;i<count;++i)if(step.begin<knots[i].local&&knots[i].local<step.begin+step.dt)
        step.dt=ugkp::boundedTimeDuration(knots[i].local,step.begin);
}
CHMT_HD inline void limitMaterialStep(ResidentMaterialStep& step,ugkp::CompensatedTime total,
    Real transportLimit,int transportError){
    if(!step.active||step.error!=MaterialStepSelectionError::None)return;
    if(transportError){step.error=MaterialStepSelectionError::Transport;return;}
    // std::min's first-operand NaN behavior is part of the original order.
    if(transportLimit<step.dt)step.dt=transportLimit;
    step.end=step.begin+step.dt;
    if(!finite(step.dt)||!(step.dt>0)||step.end==step.begin){step.error=MaterialStepSelectionError::NonAdvancing;return;}
    step.finalStep=step.end==total;
}
}
