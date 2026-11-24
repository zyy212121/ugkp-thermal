#pragma once
#include "gpu/CoupledGasInterval.H"
#include "../../../common/ParticleResidentInterval.H"
namespace chmt {
enum class GasWindowStop : int {Running,HistoryCapacity,Complete,Fatal};
struct GasWindowControls {
    Real cap=0,maximum=0,fixedDt=0,cfl=0,growth=0;
    bool adaptive=true,moving=false;
    int courantPeriod=1;
    ugkp::CompensatedTime total{};
    std::uint64_t gasGeometry=0,solidGeometry=0,acceptedStep=0;
};
struct GasWindowProgress {
    ugkp::CompensatedTime elapsed{},lastBegin{};
    std::uint64_t gasSteps=0,microSteps=0;
    std::size_t batchFrames=0,batchCapacity=0;
    Real seed=0,rate=0,requested=0,remaining=0,reference=0,dt=0,nextDt=0;
    bool courantReady=false,refreshed=false;
    GasWindowStop stop=GasWindowStop::Running;
};
struct GasWindowCourant {double value=0;int unavailable=0;};
}
