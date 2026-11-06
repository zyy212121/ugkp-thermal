#ifndef CHMT_COUPLED_GAS_INTERVAL_H
#define CHMT_COUPLED_GAS_INTERVAL_H
#include "gpu/ResidentWallProgram.H"
#include "../../../common/gasTransport/GasResidentInterval.H"
namespace chmt {
// Stage positions are sampled from the resident local clock before gas replay.
// The published absolute stamps are diagnostic/output metadata.
struct CoupledGasInterval {
    GasWallProgramPosition position[3]{};
    Real time[3]{};
    GasWallProgramPosition endpoint{};
    std::uint64_t sequence=0,geometry=0;
};
CHMT_HD inline GasWallProgramPosition coupledGasPosition(const CoupledGasInterval* interval,
    int stage,GasWallProgramPosition supplied){
    return interval?interval->position[stage]:supplied;
}
CHMT_HD inline CoupledGasInterval prepareCoupledWallInterval(ResidentWallClock& clock,
    const ResidentWallKnotTime* knots,int count,Real dt,std::uint64_t geometry){
    CoupledGasInterval interval;interval.sequence=++clock.sequence;interval.geometry=geometry;
    for(int stage=0;stage<3;++stage){
        const auto time=clock.elapsed+(stage==0?0:stage==1?dt:.5*dt);
        interval.position[stage]=locateResidentGasWallProgram(knots,count,clock.origin,time);
        interval.time[stage]=ugkp::absoluteTime(clock.origin,time);
    }
    clock.elapsed.add(dt);interval.endpoint=locateResidentGasWallProgram(knots,count,clock.origin,clock.elapsed);
    return interval;
}
#if defined(__CUDACC__)
__global__ void advanceCoupledWallClock(ResidentWallClock* clock,const ResidentWallKnotTime* knots,
    int count,Real dt,std::uint64_t geometry,CoupledGasInterval* output){
    if(!blockIdx.x&&!threadIdx.x)*output=prepareCoupledWallInterval(*clock,knots,count,dt,geometry);
}
#endif
}
#endif
