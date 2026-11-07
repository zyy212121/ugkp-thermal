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
    // Contiguous fields bind directly to gas/material conservative operators.
    // They are published by a device kernel, never through a host mesh mirror.
    const double* oldVolume=nullptr;
    const double* newVolume=nullptr;
    const double* sweptVolume=nullptr;
};
// One full native linear-trajectory geometry transaction. Inputs and controls
// are borrowed device arrays. Immutable topology, periodic pairing and the
// initial Empty extrusion are validated by the initializing mesh owner. The
// old coordinates must be an accepted state in that same topology/extrusion.
// No physical state is committed here: a CFL or nonlinear trial may evaluate
// several endpoints before its owner accepts one. All numerical decisions,
// warning counts/samples, intermediate geometry checks and GCL stay on device.
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
