#pragma once
#include "dictionary.H"
#include "scalar.H"
#include "label.H"
namespace ugkp { namespace thermal {
struct CouplingSchedule {Foam::scalar interval;Foam::label radiationFrequency;bool radiationEnabled;};
inline CouplingSchedule readCouplingSchedule(const Foam::dictionary& coupling) {
    const auto legacy=coupling.lookup<Foam::scalar>("couplingInterval");
    return {coupling.lookupOrDefault<Foam::scalar>("solidContactCouplingInterval",legacy),
        coupling.lookupOrDefault<Foam::label>("radiationCouplingFrequency",1),
        coupling.lookupOrDefault<bool>("radiationEnabled",true)};
}
inline void readRadiationSchedule(CouplingSchedule& schedule,const Foam::dictionary& radiation){
    // Same public radiationProperties authority used by CHT's resolved input.
    schedule.radiationEnabled=radiation.lookupOrDefault<bool>("enabled",false);
    schedule.radiationFrequency=radiation.lookupOrDefault<Foam::label>("couplingFrequency",1);
}
}}
