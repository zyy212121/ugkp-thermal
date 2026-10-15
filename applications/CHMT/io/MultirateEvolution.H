#ifndef CHMT_IO_MULTIRATEEVOLUTION_H
#define CHMT_IO_MULTIRATEEVOLUTION_H
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt {
struct Backend;
struct MicrostepGeometryReport {
    std::uint64_t microSequence=0;
    Real begin=0,end=0;
    std::array<HostStageGeometry,2> gas{},solid{};
};
struct WindowEvolutionReport {
    std::uint64_t acceptedGasMicrosteps=0,materialSolves=0;
    Real nextGasDt=0;
    std::vector<MicrostepGeometryReport> geometry{};
};
// Advance at most one macro interval toward targetTime. The accepted state is

// failed/replayed window never becomes output or restart authority.
bool advanceCoupledWindow(Backend&,CpuMaterialDriver&,const ModelConfig&,
    const CouplingControls&,const CpuMaterialControls&,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport&,std::string& error);
}
#endif
