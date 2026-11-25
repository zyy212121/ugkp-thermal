#ifndef CHMT_CONFIGURATION_MULTIRATEIO_H
#define CHMT_CONFIGURATION_MULTIRATEIO_H
#include "configuration/ModelIO.H"
#include "../../../common/thermal/ThermalCouplingSchedule.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt { namespace io {
enum class ExecutionMode { Multirate };
inline ExecutionMode readExecutionMode(const Foam::dictionary& d,const HostState& state) {
    (void)d;(void)state;
    return ExecutionMode::Multirate;
}


inline CouplingControls readCouplingControls(const Foam::Time& time,Real controlDeltaT,bool radiationEnabled) {
    CouplingControls controls;
    controls.scheduleOrigin=time.beginTime().value();
    controls.clock=ugkp::thermal::readOutputCouplingClock(time,radiationEnabled,controlDeltaT);
    return controls;
}

}}
#endif
