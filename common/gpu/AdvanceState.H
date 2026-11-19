#ifndef UGKP_GPU_ADVANCE_STATE_H
#define UGKP_GPU_ADVANCE_STATE_H
#include <cstdint>
#include <limits>
namespace ugkp { namespace gpu {



enum class AdvanceClock : int { physical, pseudo, sampling };
enum class AdvanceStop : int { running, boundary, stepLimit, converged, capacity, fatal };
enum class AdvanceStage : int { selectStep, gasParticle, thermal, material, geometry, accept };
constexpr unsigned advanceProgressCapacity=64;
struct AdvanceProgressPoint { std::uint64_t step; double time,dt; };
struct AdvanceProgress {
    unsigned count;
    AdvanceProgressPoint points[advanceProgressCapacity];
};

struct AdvanceRequest {
    bool watchControlFiles=false;
    double boundary=0,maximumDt=0,targetCourant=0,growthLimit=1,small=1e-15;
    double stopOffset=0,nextWrite=0,endTime=0;
    std::uint64_t maximumSteps=0,maximumThermalEvents=0;
    int courantInterval=1;
    unsigned progressInterval=0;
    bool adjustTimeStep=false,exactBoundary=true,adjustWrite=false;
    double thermalOutputTime=std::numeric_limits<double>::infinity();
    std::uint64_t thermalOutputStep=0,thermalOutputEvent=0;
    double scheduleOrigin=0,nextHeat=std::numeric_limits<double>::infinity();
};
struct AdvanceState {
    double time=0,dt=0,previousDt=0,previousTime=0,intervalEnd=0,lastCourant=0,boundaryTime=0,samplingTime=0;
    double clockDt=1,previousClockDt=1;
    double nextHeat=0,nextRadiation=0,outputBoundary=0,endTime=0;
    std::uint64_t steps=0,submissionSteps=0,acceptedSequence=0,positionSequence=0;
    std::uint64_t geometryVersion=0,thermalSequence=0,windowSequence=0;
    std::uint64_t nextProgressStep=0;
    unsigned progressCount=0;
    int stepsSinceCourant=0;
    AdvanceClock clock=AdvanceClock::physical;
    AdvanceStop stop=AdvanceStop::running;
    AdvanceStage stage=AdvanceStage::selectStep;
    bool boundaryFrozen=false,capacityResuming=false;
    std::uint64_t submissionThermalEvents=0,submissionRadiationEvents=0;
    std::uint64_t initialSteps=0,windowSamples=0,windows=0,stableWindows=0;
    double windowDuration=0,maximumRelative=0,maximumMeanChange=0,maximumInventoryDrift=0;
    bool residualReady=false,windowReady=false,stationary=false;
};
struct SteadyAdvanceControls {
    double courant=0,maximumDt=0,relativeTolerance=0,absoluteTolerance=0;
    double meanTolerance=0,inventoryTolerance=0;
    std::uint64_t residualInterval=1,minimumIterations=1,windowSamples=1,minimumWindows=1,stableWindows=1;
};


struct SnapshotStamp {
    double time=0,samplingTime=0;
    std::uint64_t steps=0,acceptedSequence=0,geometryVersion=0,thermalSequence=0,windowSequence=0;
    int precisionBits=0;
};
}}
#endif
