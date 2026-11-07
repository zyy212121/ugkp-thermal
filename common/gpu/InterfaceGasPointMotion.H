#ifndef UGKP_GPU_INTERFACE_GAS_POINT_MOTION_H
#define UGKP_GPU_INTERFACE_GAS_POINT_MOTION_H
#include "NormalHeightMotion.H"
#include "InterfacePointDisplacement.H"
#include "ExecutionGraph.H"
namespace ugkp { namespace gpu { namespace geometry {
enum class InterfaceGasMotionFailure:int {None,EmptyFreePoint,EmptyNormal,EmptyPair,NonfiniteFixed,IsolatedPoint,NoAnchor,Harmonic};
struct InterfaceGasMotionStatus {
    int changed=0,index=-1;
    InterfaceGasMotionFailure failure=InterfaceGasMotionFailure::None;
    HarmonicDeviceStatus harmonic;
    NormalHeightMotionControl constraints;
};
struct InterfaceGasPointMotionDefinition {
    NormalHeightPointMotionDefinition gas;
    Topology solidTopology;
    std::vector<Point> oldSolid;
    std::vector<int> solidFaces,solidToGas,emptyMate;
    std::vector<double> pointThickness,previousFilmThickness,filmThickness;
    int emptyAxis=-1;
};
struct InterfaceGasPointMotionInput {
    InterfacePointDisplacementInput interface;
    const Point* gasReference=nullptr;
    double geometricTolerance=0;
};
struct InterfaceGasPointMotionView {
    const Point* points=nullptr;
    const InterfaceGasMotionStatus* status=nullptr;
};
struct InterfaceGasPointMotionResult {std::vector<Point> points;InterfaceGasMotionStatus status;};
// Ordered conformal displacement, unchanged-offset decision, prescribed/Empty
// checks and relative-height harmonic motion form one resident graph. The same
// fixed mask is shared by validation and the harmonic operator. Stage geometry
// certification and physical acceptance remain with the consuming mesh owner.
class InterfaceGasPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit InterfaceGasPointMotion(const InterfaceGasPointMotionDefinition&);
    ~InterfaceGasPointMotion();
    InterfaceGasPointMotionView view()const;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,const InterfaceGasPointMotionInput*);
    void solveDevice(const InterfaceGasPointMotionInput*,void* stream=nullptr);
    InterfaceGasPointMotionResult solve(const std::vector<Point>& solidEndpoint,const std::vector<Point>& solidArea);
};
}}}
#endif
