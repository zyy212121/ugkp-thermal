#ifndef UGKP_GPU_RESIDENT_STAGE_GEOMETRY_H
#define UGKP_GPU_RESIDENT_STAGE_GEOMETRY_H
#include "PolyhedralGeometry.H"
namespace ugkp {namespace diagnostics {struct DeviceContext;} namespace gpu {namespace geometry {
enum class StageGeometryFailure : int {
    None,InvalidTolerances,InvalidInterval,NonfiniteVertex,DegenerateFace,NonfiniteCellClosure,
    NonpositiveVolume,NonfiniteCentroid,PeriodicArea,PeriodicVertices,EmptyDuplicate,
    NonfiniteWallTriangle,InvalidWallDistance,EmptyNormal,EmptyTangential,ZeroRadius,
    NonfiniteDeterminant,CollapsedFan,NonStarVolume,InteriorInversion,NonfiniteSweep,NonfiniteStageClosure
};
enum class StageGeometryPhase : int {Input,Old,EmptyTrajectory,Endpoint,LinearTrajectory,Midpoint,OneThird,TwoThird,Volume,Sweep,Conservation,PeriodicSweep};
struct StageGeometryControls {double interval=0,absoluteTolerance=0,relativeTolerance=0;};
struct StageGeometryStatus {StageGeometryFailure failure=StageGeometryFailure::None;StageGeometryPhase phase=StageGeometryPhase::Input;int index=-1;};
struct StageGeometryDeviceView {
    PolyhedralGeometryDeviceView oldGeometry{},endpointGeometry{};
    const Point* meshVelocity=nullptr;
    const StageGeometryStatus* status=nullptr;


    const double* oldVolume=nullptr;
    const double* newVolume=nullptr;
    const double* sweptVolume=nullptr;
};







class ResidentStageGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentStageGeometry(const Topology&,int pointCount,const std::vector<int>& walls,
        const std::vector<int>& periodicPartner,int emptyAxis,const std::vector<int>& emptyMate);
    ~ResidentStageGeometry();
    StageGeometryDeviceView view()const;
    void prepare(const Point* old,const Point* endpoint,const StageGeometryControls* controls,
        ugkp::diagnostics::DeviceContext* diagnostics=nullptr,void* stream=nullptr);
};
}}}
#endif
