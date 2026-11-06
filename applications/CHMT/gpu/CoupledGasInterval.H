#ifndef CHMT_COUPLED_GAS_INTERVAL_H
#define CHMT_COUPLED_GAS_INTERVAL_H
#include "gpu/GasWindowProgram.H"
#include "../../../common/gasTransport/GasResidentInterval.H"
namespace chmt {
// The host keeps WindowTime interpolation semantics. Only the compact stage
// positions and record identity change between replays of the shared gas graph.
struct CoupledGasInterval {
    GasWallProgramPosition position[3]{};
    Real time[3]{};
    std::uint64_t sequence=0,geometry=0;
};
CHMT_HD inline GasWallProgramPosition coupledGasPosition(const CoupledGasInterval* interval,
    int stage,GasWallProgramPosition supplied){
    return interval?interval->position[stage]:supplied;
}
}
#endif
