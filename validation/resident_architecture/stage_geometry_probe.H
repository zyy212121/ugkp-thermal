#pragma once
#include "common/gpu/ResidentStageGeometry.H"
#include "common/Diagnostics.H"
namespace stageProbe {
using namespace ugkp::gpu::geometry;
struct Result {
    StageGeometryStatus status{};ugkp::diagnostics::DeviceContext diagnostics{};
    std::vector<FaceResult> faces;std::vector<CellResult> cells;std::vector<WallDistanceResult> distances;
    std::vector<SweepResult> sweep;std::vector<Point> velocity;
};
Result evaluate(const Topology&,const std::vector<Point>&,const std::vector<Point>&,const std::vector<int>&,
    const std::vector<int>&,int,const std::vector<int>&,StageGeometryControls);
}
