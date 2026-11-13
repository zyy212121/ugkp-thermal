#ifndef CHMT_CONFIGURATION_MULTIRATEIO_H
#define CHMT_CONFIGURATION_MULTIRATEIO_H
#include "configuration/ModelIO.H"
#include "FoamSolverScheme.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt { namespace io {
enum class ExecutionMode { Multirate };
inline ExecutionMode readExecutionMode(const Foam::dictionary& d,const HostState& state) {
    (void)d;(void)state;
    return ExecutionMode::Multirate;
}

inline CouplingControls readCouplingControls(const Foam::Time& time, Real controlDeltaT, Real thermalInterval=0) {
    const auto controlsDictionary=ugkp::solverControls::startupScheme(time).section("thermalCoupling");
    require(thermalInterval>0||controlsDictionary.found("couplingInterval"),
        "thermalCoupling.couplingInterval is required for executionMode Multirate");
    CouplingControls controls;
    controls.interval = thermalInterval>0?thermalInterval:readNumber(controlsDictionary,"couplingInterval");
    // deltaT is a gas cap only. Neither it nor a material local step limits H.
    (void)controlDeltaT; // Gas numerical caps are supplied by constant/scheme.
    // Numerical material steps are configured only by constant/scheme.
    return controls;
}


}}
#endif
