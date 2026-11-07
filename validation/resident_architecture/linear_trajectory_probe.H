#pragma once
#include "common/gpu/PolyhedralGeometry.H"
#include "common/gpu/LinearTrajectoryMath.H"
#include <vector>
namespace trajectoryProbe {
using namespace ugkp::gpu::geometry;
struct Report {TrajectoryFailure failure=TrajectoryFailure::None;int warnings=0;double observed=0,tolerance=0;};
std::vector<Report> evaluate(const Topology&,const std::vector<Point>&,const std::vector<Point>&);
}
