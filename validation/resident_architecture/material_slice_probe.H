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
};
struct Result {
    ugkp::IntervalSlicePlan plan{};
    ugkp::IntervalSliceCursor cursor{};
    std::vector<chmt::ExchangePacket> packets;
    std::vector<double> radiation;
    std::vector<chmt::SolidQ> inventory;
    std::vector<int> inventoryStatus;
    int bindingParity=0;
};
std::vector<Result> evaluate(const Test&);
}
