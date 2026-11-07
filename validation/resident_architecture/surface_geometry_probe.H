#pragma once
#include "common/gpu/ResidentStageGeometry.H"
#include "common/gpu/ResidentSurfaceGeometry.H"
#include "common/Diagnostics.H"
namespace surfaceProbe {
using namespace ugkp::gpu::geometry;
struct Mesh {Topology topology;std::vector<Point> old,end;std::vector<int> walls,partner,mate;int axis=-1;};
struct Result {SurfaceGeometryStatus status{};std::vector<SurfaceFaceGeometry> faces;std::vector<EdgeResult> edges;ugkp::diagnostics::DeviceContext diagnostics{};};
Result evaluate(const Mesh&,const Mesh&,const std::vector<SurfaceFaceMap>&,const std::vector<SurfaceEdgeMap>&,double);
}
