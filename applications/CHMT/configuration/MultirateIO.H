#ifndef CHMT_CONFIGURATION_MULTIRATEIO_H
#define CHMT_CONFIGURATION_MULTIRATEIO_H
#include "configuration/ModelIO.H"
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt { namespace io {
enum class ExecutionMode { Multirate, LegacyExplicit, Standalone };

inline ExecutionMode readExecutionMode(const Foam::dictionary& d, const HostState& state) {
    // These pre-existing declarations explicitly select independent verification
    // models, so their operator cases need no material-coupling dictionary.
    const bool declaredStandalone = d.found("normalColumn")
        || (d.found("surface") && flag(d.subDict("surface"), "standalone"))
        || (state.solid.empty() && state.surface.area.empty());
    const std::string name = text(d, "executionMode", declaredStandalone?"Standalone":"Multirate");
    if (name == "Multirate") return ExecutionMode::Multirate;
    if (name == "LegacyExplicit") return ExecutionMode::LegacyExplicit;
    if (name == "Standalone") return ExecutionMode::Standalone;
    throw std::runtime_error("executionMode must be Multirate, LegacyExplicit or Standalone");
}

inline CouplingControls readCouplingControls(const Foam::dictionary& d, Real controlDeltaT) {
    require(finite(controlDeltaT) && controlDeltaT>0, "controlDict deltaT must be positive and finite");
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
    require(controls.interval>0 && controls.minimumInterval>0 && controls.minimumInterval<=controls.interval,
        "multirate coupling intervals must satisfy 0 < minimumCouplingInterval <= couplingInterval");
    require(controls.gasMaxDt>0 && controls.materialMaxSubstep>=0,
        "multirate gasMaxDt must be positive; materialMaxSubstep must be nonnegative (zero means no accuracy cap)");
    require(controls.maxWindowIterations>0 && controls.maxWindowRetries>=0 && controls.predictionTolerance>0,
        "invalid multirate iteration, retry or prediction tolerance");
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
    require(controls.transportCfl>0 && controls.transportCfl<=1 && controls.reactionCfl>0
        && controls.reactionCfl<=1 && controls.reactionMaxSubstep>=0 && controls.maxSubsteps>0
        && controls.nonlinearMaxIterations>0 && controls.nonlinearRelativeTolerance>0
        && controls.energyAbsoluteTolerance>=0 && controls.energyRelativeTolerance>=0
        && controls.driveRelativeTolerance>=0 && controls.driveSourceFraction>0
        && controls.driveSourceFraction<=1 && controls.driveTractionScale>0,
        "invalid cpuMaterial controls");
    return controls;
}
}}
#endif
