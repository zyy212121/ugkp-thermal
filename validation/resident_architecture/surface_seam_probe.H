#pragma once
#include "common/gpu/ResidentSurfaceSeams.H"
namespace seamProbe {
using namespace ugkp::gpu::geometry;
struct Sample {std::vector<Point> points;std::vector<FaceResult> faces;};
struct Result {SurfaceSeamStatus status;SurfaceGeometryStatus surfaceStatus;
    std::vector<SurfaceEdgeMap> edges;std::vector<SurfaceFaceGeometry> faceMetrics;std::vector<EdgeResult> edgeMetrics;};
std::vector<Result> evaluate(const Topology&,const std::vector<SurfaceFaceMap>&,const std::vector<int>& surfaceFaces,const std::vector<SurfaceTopologyEdge>&,
    const std::vector<SurfaceSeamPatch>&,const std::vector<Sample>&);
}
