#ifndef UGKP_GPU_RESIDENT_SURFACE_GEOMETRY_H
#define UGKP_GPU_RESIDENT_SURFACE_GEOMETRY_H
#include "PolyhedralGeometry.H"
namespace ugkp {namespace diagnostics {struct DeviceContext;} namespace gpu {namespace geometry {
struct SurfaceSeamStatus;
struct SurfaceSeamDeviceView;
struct SurfaceFaceMap {int solidFace=-1,gasFace=-1,solidCell=-1,gasCell=-1;};
struct SurfaceEdgeMap {SurfaceEdge geometry{};int ownerSurface=-1;};
struct SurfaceFaceGeometry {
    double oldArea=0,area=0,solidDistance=0,gasDistance=0;
    Point centre{},normal{},velocity{};
};
enum class SurfaceGeometryFailure : int {None,InvalidInterval,InvalidNormalDistance,DegenerateConormal,DegenerateHinge,AmbiguousSeam,UnpairedSeam};
struct SurfaceGeometryStatus {
    unsigned long long first=0;int firstWarning=0,index=-1,edgeCount=0;
    SurfaceGeometryFailure failure=SurfaceGeometryFailure::None;
    double scale=0;
};
struct SurfaceGeometryDeviceView {
    int faces=0,edges=0;
    const SurfaceFaceGeometry* face=nullptr;
    const EdgeResult* edge=nullptr;
    const SurfaceGeometryStatus* status=nullptr;
};
// The coupled owner supplies already certified gas/solid stage geometry. This
// operator computes only the interface metrics and adaptive edge quadrature;
// it does not re-run either region's completed geometry transaction. Face
// correspondence is immutable. Edge maps may be produced/updated on device.
class ResidentSurfaceGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;
    void prepareImpl(PolyhedralGeometryDeviceView,PolyhedralGeometryDeviceView,PolyhedralGeometryDeviceView,
        const Point*,const Point*,const SurfaceEdgeMap*,const double*,const SurfaceSeamStatus*,
        ugkp::diagnostics::DeviceContext*,void*);
public:
    ResidentSurfaceGeometry(const std::vector<SurfaceFaceMap>& faces,int edges,int solidPoints,bool emptySolid);
    ~ResidentSurfaceGeometry();
    SurfaceGeometryDeviceView view()const;
    void prepare(PolyhedralGeometryDeviceView oldSolid,PolyhedralGeometryDeviceView solid,
        PolyhedralGeometryDeviceView gas,const Point* oldPoints,const Point* points,
        const SurfaceEdgeMap* edges,const double* interval,
        ugkp::diagnostics::DeviceContext* diagnostics=nullptr,void* stream=nullptr);
    // The seam owner supplies the compacted map and its active edge count on
    // device. Invalid seam candidates retain native face-before-edge ordering.
    void prepareWithSeams(PolyhedralGeometryDeviceView oldSolid,PolyhedralGeometryDeviceView solid,
        PolyhedralGeometryDeviceView gas,const Point* oldPoints,const Point* points,
        SurfaceSeamDeviceView seams,const double* interval,
        ugkp::diagnostics::DeviceContext* diagnostics=nullptr,void* stream=nullptr);
};
}}}
#endif
