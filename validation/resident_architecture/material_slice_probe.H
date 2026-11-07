#pragma once
#include "applications/CHMT/gpu/ResidentMaterialSlice.H"
#include <vector>
namespace materialSliceProbe {
struct Test {
    std::size_t faces=0;
    std::vector<chmt::ResidentMaterialDriveInterval> intervals;
    std::vector<chmt::CoupledSurfaceDownload> surface;
    std::vector<double> external;
    std::vector<ugkp::CompensatedTime> targets;
    std::vector<int> accepted;
    ugkp::CompensatedTime total{};
    bool particles=false;
    int recessionCase=0;
};
struct Result {
    ugkp::IntervalSlicePlan plan{};
    ugkp::IntervalSliceCursor cursor{};
    std::vector<chmt::ExchangePacket> packets;
    std::vector<double> radiation;
    std::vector<chmt::SolidQ> inventory;
    std::vector<int> inventoryStatus;
    int bindingParity=0;
    std::vector<double> recession;
    std::vector<chmt::FilmAux> fronts;
    std::vector<int> recessionStatus;
};
inline chmt::PhysicsConfig recessionPhysics(int mode){
    chmt::PhysicsConfig p;
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.Tmin=100;c.Tmax=3000;}
    for(auto& c:p.species){c.R=287;c.cp0=1000;c.Tmin=100;c.Tmax=3000;}
    if(mode==8)p.condensed[1].rho=0;
    return p;
}
inline std::vector<chmt::SolidQ> recessionInventory(int mode){
    std::vector<chmt::SolidQ> q(3);for(int c=0;c<3;++c){q[c].condensed[0]=.2+.1*c;q[c].energy=q[c].condensed[0]*1000*700;}
    if(mode==7)q[1].energy=-1;return q;
}
inline chmt::FilmAux recessionFront(std::size_t face){
    chmt::FilmAux a;a.solidFront=.001*face;a.gasFront=.003*face;a.temperature=601+face;
    a.normalVelocity=.01;a.solidNormalVelocity=.02;a.pressure=123+face;return a;
}
std::vector<Result> evaluate(const Test&);
}
