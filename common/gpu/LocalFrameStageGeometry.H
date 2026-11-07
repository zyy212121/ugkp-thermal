#ifndef UGKP_GPU_LOCAL_FRAME_STAGE_GEOMETRY_H
#define UGKP_GPU_LOCAL_FRAME_STAGE_GEOMETRY_H
#include "ResidentStageGeometry.H"
#include "ExecutionGraph.H"
#include "../Diagnostics.H"
namespace ugkp { namespace gpu { namespace geometry {
struct LocalFrameStageDefinition {
    Topology topology;
    std::vector<Point> worldOld,localOld;
    std::vector<int> walls,periodicPartner,emptyMate;
    int emptyAxis=-1;
};
struct LocalFrameStageInput {
    const Point* worldOld=nullptr;
    const Point* localOld=nullptr;
    const Point* localEndpoint=nullptr;
    StageGeometryControls controls;
};
struct LocalFrameStageView {
    const Point* points=nullptr;
    StageGeometryDeviceView geometry;
    const ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
};
struct LocalFrameStageResult {
    std::vector<Point> points;
    StageGeometryStatus status;
    ugkp::diagnostics::DeviceContext diagnostics{};
    RebuildResult endpoint;
    std::vector<double> oldVolume,sweep;
    std::vector<Point> velocity;
};
// Converts a resident local-frame trial to its representable world-coordinate
// trajectory and certifies it. All geometry and diagnostic state is trial-local.
class LocalFrameStageGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit LocalFrameStageGeometry(const LocalFrameStageDefinition&);
    ~LocalFrameStageGeometry();
    LocalFrameStageView view()const;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,const LocalFrameStageInput*);
    void prepare(const LocalFrameStageInput*,void* stream=nullptr);
    LocalFrameStageResult solveBorrowed(const Point* localEndpoint,StageGeometryControls);
};
}}}
#endif
