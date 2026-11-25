#pragma once
#include "dictionary.H"
#include "scalar.H"
#include "label.H"
#include "../FoamSolverScheme.H"
#include "../gpu/PhysicalEventClock.H"
namespace ugkp { namespace thermal {
inline gpu::OutputCouplingClock readOutputCouplingClock
(
    const Foam::Time& time,bool radiationEnabled,double zeroContactPeriod=0
)
{
    const auto& scheme=ugkp::solverControls::startupScheme(time);
    Foam::scalar contact=1,radiation=1;
    scheme.read("thermalCoupling","solidContactCouplingInterval",contact);
    scheme.read("thermalCoupling","radiationCouplingFrequency",radiation);
    const auto& controls=time.controlDict();
    const auto mode=controls.lookupOrDefault<Foam::word>("writeControl","timeStep");
    const auto write=controls.lookupOrDefault<Foam::scalar>("writeInterval",1);
    const double outputPeriod=(mode=="runTime"||mode=="adjustableRunTime") && write>0
        ?time.userTimeToTime(write):0;
    auto clock=gpu::makeOutputCouplingClock(outputPeriod,contact,radiation,radiationEnabled);
    if(clock.contactPeriod==0 && zeroContactPeriod>0){
        clock.contactPeriod=zeroContactPeriod;
        clock.radiationPeriod=radiationEnabled?zeroContactPeriod*gpu::couplingFraction(radiation):0;
    }
    return clock;
}

}}
