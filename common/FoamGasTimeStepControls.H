#pragma once
#include "FoamSolverScheme.H"
#include "GpuSchedulingConfiguration.H"
namespace ugkp { namespace solverControls {
struct GasTimeStepControls {
    bool adjustTimeStep=false;
    Foam::scalar maxCo=.5,maxDeltaT=Foam::GREAT;
    Foam::label courantUpdateInterval=Foam::GpuSchedulingConfiguration{}.courantUpdateInterval;
    Foam::scalar maximumTimeStepGrowth=Foam::GpuSchedulingConfiguration{}.maxDeltaTGrowth;
};
inline GasTimeStepControls readGasTimeStepControls(const Foam::Time& time,
    const Foam::word& region=Foam::word::null) {
    GasTimeStepControls c;const auto& scheme=startupScheme(time);
#define UGKP_GAS_TIME(Key) scheme.read("gasTimeStep",#Key,c.Key,"effectiveDefault",region)
    UGKP_GAS_TIME(adjustTimeStep);UGKP_GAS_TIME(maxCo);UGKP_GAS_TIME(maxDeltaT);
    UGKP_GAS_TIME(courantUpdateInterval);UGKP_GAS_TIME(maximumTimeStepGrowth);
#undef UGKP_GAS_TIME
    return c;
}
}}
