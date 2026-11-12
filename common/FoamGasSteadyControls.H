#pragma once
#include "FoamSolverScheme.H"
#include "scalar.H"
#include <type_traits>
namespace ugkp { namespace solverControls {
using SteadySolverReal=typename std::conditional<ugkp::precision::compiledBits==32,float,Foam::scalar>::type;
struct GasSteadyControls {
    SteadySolverReal maxCo=SteadySolverReal(.5);
    Foam::scalar maxDeltaT=Foam::GREAT,boundaryTime=0,referenceTime=1;
    SteadySolverReal relativeTolerance=sizeof(SteadySolverReal)==4?SteadySolverReal(1e-4):SteadySolverReal(1e-8);
    SteadySolverReal absoluteTolerance=0;
    Foam::label residualInterval=10,minimumIterations=1;
};
struct ParticleSteadyControls {
    Foam::scalar samplingDeltaT=0,boundaryTime=0;
    Foam::label windowSamples=200,minimumWindows=4,stableWindows=2;
    SteadySolverReal meanTolerance=SteadySolverReal(1e-3),inventoryTolerance=SteadySolverReal(1e-2);
};
// Preserve the existing count normalization; it is not a convergence tolerance
// floor. The reported value is the value actually used by the modulo/window.
inline void readSteadyCount(const Scheme& scheme,const char* module,const char* key,
    Foam::label& value,Foam::label minimum,const Foam::word& region) {
    const auto section=scheme.section(module,region);
    const bool supplied=section.readIfPresent(key,value,false,false);
    const Foam::label requested=value;
    value=Foam::max(minimum,value);
    Foam::Info<<"SCHEME_COUNT module="<<module<<" key="<<key<<" requested="<<requested
        <<" effective="<<value<<" existingMinimum="<<minimum<<Foam::nl;
    scheme.report(module,key,value,supplied,"effectiveDefault",region);
}
inline GasSteadyControls readGasSteadyControls(const Foam::Time& time,
    Foam::scalar restartBoundaryTime,const Foam::word& region=Foam::word::null) {
    GasSteadyControls c;c.boundaryTime=restartBoundaryTime;
    const auto& scheme=startupScheme(time);
#define UGKP_STEADY_GAS(Key) scheme.read("steadyGas",#Key,c.Key,"precisionDefault/restartClock",region)
    UGKP_STEADY_GAS(maxCo);UGKP_STEADY_GAS(maxDeltaT);UGKP_STEADY_GAS(boundaryTime);
    UGKP_STEADY_GAS(relativeTolerance);UGKP_STEADY_GAS(absoluteTolerance);
    UGKP_STEADY_GAS(minimumIterations);UGKP_STEADY_GAS(referenceTime);
#undef UGKP_STEADY_GAS
    readSteadyCount(scheme,"steadyGas","residualInterval",c.residualInterval,1,region);
    return c;
}
inline ParticleSteadyControls readParticleSteadyControls(const Foam::Time& time,
    Foam::scalar defaultDeltaT,Foam::scalar restartBoundaryTime,
    const Foam::word& region=Foam::word::null) {
    ParticleSteadyControls c;c.samplingDeltaT=defaultDeltaT;c.boundaryTime=restartBoundaryTime;
    const auto& scheme=startupScheme(time);
#define UGKP_STEADY_PARTICLE(Key) scheme.read("steadyParticle",#Key,c.Key,"effectiveDefault/restartClock",region)
    UGKP_STEADY_PARTICLE(samplingDeltaT);UGKP_STEADY_PARTICLE(boundaryTime);
    UGKP_STEADY_PARTICLE(meanTolerance);UGKP_STEADY_PARTICLE(inventoryTolerance);
#undef UGKP_STEADY_PARTICLE
    readSteadyCount(scheme,"steadyParticle","windowSamples",c.windowSamples,1,region);
    readSteadyCount(scheme,"steadyParticle","minimumWindows",c.minimumWindows,2,region);
    readSteadyCount(scheme,"steadyParticle","stableWindows",c.stableWindows,1,region);
    return c;
}
}}
