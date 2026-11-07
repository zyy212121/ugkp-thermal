#pragma once
#include "applications/CHMT/gpu/ResidentMaterialStep.H"
#include "applications/CHMT/materials/CpuMaterialDriver.H"
#include <vector>
namespace materialStepProbe {
struct Test {
    ugkp::CompensatedTime begin{},total{2};
    chmt::CpuMaterialControls controls{};
    std::vector<chmt::ResidentWallKnotTime> knots;
    chmt::Real transportLimit=1;
    int transportError=0;
    bool changed=false;
};
std::vector<chmt::ResidentMaterialStep> evaluate(const std::vector<Test>&);
}
