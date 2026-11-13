#pragma once
#include "dictionary.H"
#include "scalar.H"
#include "label.H"
#include "../FoamSolverScheme.H"
namespace ugkp { namespace thermal {
struct CouplingSchedule {Foam::scalar interval;Foam::label radiationFrequency;bool radiationEnabled;};
inline CouplingSchedule readCouplingSchedule(const Foam::dictionary& coupling,const Foam::Time& time) {
    const auto& scheme=ugkp::solverControls::startupScheme(time);
    const auto controls=scheme.section("thermalCoupling");
    // Coupling interval was required before migration; retain that contract.
    const auto base=controls.lookup<Foam::scalar>("couplingInterval");
    Foam::scalar interval=base;Foam::label frequency=1;
    scheme.read("thermalCoupling","solidContactCouplingInterval",interval,"couplingInterval");
    scheme.read("thermalCoupling","radiationCouplingFrequency",frequency);
    return {interval,frequency,coupling.lookupOrDefault<bool>("radiationEnabled",true)};
}
inline void readRadiationSchedule(CouplingSchedule& schedule,const Foam::dictionary& radiation,const Foam::Time& time){
    schedule.radiationEnabled=radiation.lookupOrDefault<bool>("enabled",false);
    schedule.radiationFrequency=1;
    ugkp::solverControls::startupScheme(time).read("thermalCoupling","radiationCouplingFrequency",schedule.radiationFrequency);
}
}}
