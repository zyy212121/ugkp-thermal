#pragma once
#include "gpu/ResidentMaterialDrive.H"
#include <vector>
#include "common/gasTransport/GasDriveSelection.H"
namespace gasDriveProbe {
struct Pair {
    std::vector<chmt::CoupledSurfaceDownload> held,sample;
    std::vector<chmt::CoupledWallDownload> heldWall,sampleWall;
    double tractionScale=1;
};
std::vector<ugkwp::GasDriveComparison> evaluate(const std::vector<Pair>&);
struct SelectionTest {
    std::size_t pair=0,frames=2;
    ugkwp::GasDriveControls controls;
    chmt::ResidentMaterialDriveInterval intervals[2]{{ugkp::CompensatedTime(0),ugkp::CompensatedTime(1),1},
        {ugkp::CompensatedTime(1),ugkp::CompensatedTime(2),1}};
};
std::vector<ugkwp::GasDriveSelection> select(const std::vector<Pair>&,const std::vector<SelectionTest>&);
}
