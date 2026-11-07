#pragma once
#include "ResidentSurfaceGeometry.H"
#include "SurfaceSeamMath.H"
namespace ugkp { namespace gpu { namespace geometry {
struct SurfaceSeamPatch {int face=-1,partner=-1;};
enum class SurfaceSeamFailure : int {None,Ambiguous,Unpaired};
struct SurfaceSeamStatus {
    int count=0,index=-1;SurfaceSeamFailure failure=SurfaceSeamFailure::None;double scale=0;
};
struct SurfaceSeamDeviceView {
    int capacity=0;const SurfaceEdgeMap* edges=nullptr;const SurfaceSeamStatus* status=nullptr;
};
// Raw, non-periodic surface adjacency and the first incident cyclic patch per
// edge depend only on topology and are installed once. Geometric seam pairing
// retains the native greedy order, ambiguity checks and scale-relative tolerance
// on each update. Candidate searches within that order run cooperatively.
class ResidentSurfaceSeams {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentSurfaceSeams(int pointCount,int meshFaces,const std::vector<int>& surfaceFaces,
        const std::vector<SurfaceTopologyEdge>& rawEdges,const std::vector<SurfaceSeamPatch>& patches);
    ~ResidentSurfaceSeams();
    SurfaceSeamDeviceView view()const;
    void prepare(const Point* points,const FaceResult* faces,void* stream=nullptr);
};
}}}
