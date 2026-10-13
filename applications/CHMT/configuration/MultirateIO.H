#ifndef CHMT_CONFIGURATION_MULTIRATEIO_H
#define CHMT_CONFIGURATION_MULTIRATEIO_H
#include "configuration/ModelIO.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt { namespace io {
enum class ExecutionMode { Multirate };
inline ExecutionMode readExecutionMode(const Foam::dictionary& d,const HostState& state) {
    (void)d;(void)state;
    return ExecutionMode::Multirate;
}

inline CouplingControls readCouplingControls(const Foam::dictionary& d, Real controlDeltaT) {
    require(d.found("multirate") && d.subDict("multirate").found("couplingInterval"),
        "multirate.couplingInterval is required for executionMode Multirate");
    const auto& m = d.subDict("multirate");
    CouplingControls controls;
    controls.interval = readNumber(m, "couplingInterval");
    controls.minimumInterval = readNumber(m, "minimumCouplingInterval", controls.interval*1e-6);
    // deltaT is a gas cap only. Neither it nor a material local step limits H.
    controls.gasMaxDt = std::min(controlDeltaT, readNumber(m, "gasMaxDt", controlDeltaT));
    controls.materialMaxSubstep = readNumber(m, "materialMaxSubstep", 0);
    controls.maxWindowIterations = m.lookupOrDefault<Foam::label>("maxWindowIterations", controls.maxWindowIterations);
    controls.maxWindowRetries = m.lookupOrDefault<Foam::label>("maxWindowRetries", controls.maxWindowRetries);
    controls.predictionTolerance = readNumber(m, "predictionTolerance", controls.predictionTolerance);
    controls.enforcePredictionAccuracy = flag(m, "enforcePredictionAccuracy", controls.enforcePredictionAccuracy);
    return controls;
}

inline CpuMaterialControls readCpuMaterialControls(const Foam::dictionary& d, const CouplingControls& coupling) {
    CpuMaterialControls controls;
    controls.maxSubstep = coupling.materialMaxSubstep;
    if (d.found("cpuMaterial")) {
        const auto& m = d.subDict("cpuMaterial");
        controls.transportCfl = readNumber(m, "transportCfl", controls.transportCfl);
        controls.reactionCfl = readNumber(m, "reactionCfl", controls.reactionCfl);
        controls.reactionMaxSubstep = readNumber(m, "reactionMaxSubstep", controls.reactionMaxSubstep);
        controls.driveRelativeTolerance = readNumber(m, "driveRelativeTolerance", controls.driveRelativeTolerance);
        controls.driveSourceFraction = readNumber(m, "driveSourceFraction", controls.driveSourceFraction);
        controls.driveTractionScale = readNumber(m, "driveTractionScale", controls.driveTractionScale);
        controls.maxSubsteps = m.lookupOrDefault<Foam::label>("maxSubsteps", controls.maxSubsteps);
        controls.nonlinearMaxIterations = m.lookupOrDefault<Foam::label>("nonlinearMaxIterations", controls.nonlinearMaxIterations);
        controls.nonlinearRelativeTolerance = readNumber(m, "nonlinearRelativeTolerance", controls.nonlinearRelativeTolerance);
        controls.energyAbsoluteTolerance = readNumber(m, "energyAbsoluteTolerance", controls.energyAbsoluteTolerance);
        controls.energyRelativeTolerance = readNumber(m, "energyRelativeTolerance", controls.energyRelativeTolerance);
    }
    return controls;
}
}}
#endif
