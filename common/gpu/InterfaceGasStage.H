#ifndef UGKP_GPU_INTERFACE_GAS_STAGE_H
#define UGKP_GPU_INTERFACE_GAS_STAGE_H
#include "InterfaceGasPointMotion.H"
#include "LocalFrameStageGeometry.H"
#include "ResidentStageGeometry.H"
#include "../Diagnostics.H"
namespace ugkp { namespace gpu { namespace geometry {
struct WetOffsetVertex {int solid=0,gas=-1;Point oldSolid{},oldGas{};};
struct WetOffsetFace {int solid=0,gas=0,first=0,end=0,interfaceIndex=0;bool oldMismatch=false;};
struct InterfaceGasStageDefinition {
    std::vector<WetOffsetFace> wetFaces;
    std::vector<WetOffsetVertex> wetVertices;
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



class InterfaceGasStage {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    InterfaceGasStageResult collect();
    const int* invalidWetOffset() const;
    ExecutionGraphNode appendBorrowed(ExecutionGraph,ExecutionGraphNode,LocalFrameStageView,StageGeometryControls);
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
