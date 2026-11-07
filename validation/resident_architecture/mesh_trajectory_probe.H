#pragma once
#include "applications/CHMT/core/HostState.H"
#include "applications/CHMT/coupling/IntervalContracts.H"
#include "common/gpu/ResidentStageGeometry.H"
namespace trajectoryProbe {
struct Step {ugkp::gpu::geometry::StageGeometryStatus status;chmt::HostMesh mesh;chmt::HostStageGeometry stage;std::vector<ugkp::gpu::geometry::Point> previous,current,endpoint;};
std::vector<Step> evaluate(const chmt::HostMesh&,const chmt::WallProgram&,bool gas,int steps,double dt);
}
