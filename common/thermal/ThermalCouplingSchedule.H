#pragma once
#include "dictionary.H"
#include "scalar.H"
#include "label.H"
#include "../FoamSolverScheme.H"
#include "../gpu/PhysicalEventClock.H"
namespace ugkp { namespace thermal {
struct CouplingSchedule {Foam::scalar interval;Foam::label radiationFrequency;bool radiationEnabled;};
inline CouplingSchedule readCouplingSchedule(const Foam::dictionary& coupling,const Foam::Time& time) {
    const auto& scheme=ugkp::solverControls::startupScheme(time);
    const auto controls=scheme.section("thermalCoupling");

    const auto base=controls.lookup<Foam::scalar>("couplingInterval");
    Foam::scalar interval=base;Foam::label frequency=1;
    scheme.read("thermalCoupling","solidContactCouplingInterval",interval,"couplingInterval");
    scheme.read("thermalCoupling","radiationCouplingFrequency",frequency);
    return {interval,frequency,coupling.lookupOrDefault<bool>("radiationEnabled",true)};
}
inline gpu::OutputCouplingClock readChtCouplingClock
(
    const Foam::Time& time,bool radiationEnabled
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
    return gpu::makeOutputCouplingClock(outputPeriod,contact,radiation,radiationEnabled);
}

inline void readRadiationSchedule(CouplingSchedule& schedule,const Foam::dictionary& radiation,const Foam::Time& time){
    schedule.radiationEnabled=radiation.lookupOrDefault<bool>("enabled",false);
    schedule.radiationFrequency=1;
    ugkp::solverControls::startupScheme(time).read("thermalCoupling","radiationCouplingFrequency",schedule.radiationFrequency);
}
}}
