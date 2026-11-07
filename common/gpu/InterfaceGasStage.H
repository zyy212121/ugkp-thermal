#ifndef UGKP_GPU_INTERFACE_GAS_STAGE_H
#define UGKP_GPU_INTERFACE_GAS_STAGE_H
#include "InterfaceGasPointMotion.H"
#include "ResidentStageGeometry.H"
#include "../Diagnostics.H"
namespace ugkp { namespace gpu { namespace geometry {
struct InterfaceGasStageDefinition {
    InterfaceGasPointMotionDefinition motion;
    std::vector<int> wallFaces,periodicPartner;
};
struct InterfaceGasStageInput {
    InterfaceGasPointMotionInput motion;
    StageGeometryControls controls;
};
struct InterfaceGasStageView {
    InterfaceGasPointMotionView motion;
    StageGeometryDeviceView geometry;
    const ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
};
struct InterfaceGasStageResult {
    InterfaceGasPointMotionResult motion;
    StageGeometryStatus status;
    ugkp::diagnostics::DeviceContext diagnostics{};
    RebuildResult endpoint;
    std::vector<double> oldVolume,sweep;
    std::vector<Point> velocity;
};
// Trial-local motion and trajectory certification. The harmonic endpoint is
// consumed directly, without host publication between the two operators.
// Failure and warning state are trial-local; the caller decides acceptance.
class InterfaceGasStage {
    struct Implementation;std::unique_ptr<Implementation> data_;
    InterfaceGasStageResult collect();
public:
    explicit InterfaceGasStage(const InterfaceGasStageDefinition&);
    ~InterfaceGasStage();
    InterfaceGasStageView view()const;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,const InterfaceGasStageInput*);
    void prepare(const InterfaceGasStageInput*,void* stream=nullptr);
    InterfaceGasStageResult solveBorrowed(const Point* solidEndpoint,const Point* solidArea,StageGeometryControls);
    InterfaceGasStageResult solve(const std::vector<Point>& solidEndpoint,
        const std::vector<Point>& solidArea,StageGeometryControls);
};
}}}
#endif
