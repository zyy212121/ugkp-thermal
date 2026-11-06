#pragma once
#include "common/gpu/PolyhedralGeometry.H"
#include <vector>
namespace boundaryProbe {
using namespace ugkp::gpu::geometry;
struct Result {GeometryBoundaryStatus periodic,trajectory,unique;};
Result evaluate(const Topology&,const std::vector<Point>&,const std::vector<Point>&,const std::vector<FaceResult>&,
    const std::vector<int>&,int,const std::vector<int>&);
}
