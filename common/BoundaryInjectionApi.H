#pragma once
#include "BoundaryInjectionSampling.H"
#include "gpu/PolyhedralGeometryMath.H"
namespace ugkwp {
struct BoundaryInjectionStorage {
    int* offsets=nullptr;
    int* vertices=nullptr;
    int* scratch=nullptr;
    int* counts=nullptr;
    ugkp::gpu::geometry::Point* points=nullptr;
    BoundaryInjectionTriangle* triangles=nullptr;
};
}
// Source-local polygon offsets and point identities, independent of the
// particle source inventory. Uploading geometry never resets residual mass.
extern "C" int ugkwpGpuResidentStrictUploadBoundarySourceGeometry
(void* handle, int nVertices, const int* offsets, const int* vertices,
 int nPoints, const ugkp::gpu::geometry::Point* points);
