#ifndef UGKP_GPU_HARMONIC_SWEEP_CORRECTION_H
#define UGKP_GPU_HARMONIC_SWEEP_CORRECTION_H
#include "PolyhedralGeometry.H"
#include "ExecutionGraph.H"
namespace ugkp { namespace gpu { namespace geometry {
struct HarmonicBoundaryMode {
    int point=-1,partner=-1;
    Point displacement{};
};
struct HarmonicSweepDefinition {
    Topology topology;
    std::vector<Point> points;
    std::vector<int> offsets,neighbours,faces;
    std::vector<unsigned char> fixed;
    std::vector<double> scales;
    std::vector<PlanarConstraint> planar;
    std::vector<HarmonicBoundaryMode> modes;
    double harmonicTolerance=0;
};
struct HarmonicSweepDeviceProblem {
    const Point* old=nullptr;
    const Point* endpoint=nullptr;
    const double* scales=nullptr;
    const PlanarConstraint* planar=nullptr;
    const HarmonicBoundaryMode* modes=nullptr;
    const double* rhs=nullptr;
    int columns=0;
    double harmonicTolerance=0,rankTolerance=0,consistencyTolerance=0;
};
struct HarmonicSweepStatus {
    int error=0,mode=0,modeSolves=0,rank=0;
    int failedMode=-1,errorPoint=-1,linearInitialized=0,linearValid=0;
    // 1: invalid input/mode; 2: harmonic solve; 3: minimum-norm solve.
};
struct HarmonicSweepDeviceView {
    const Point* correction=nullptr;
    const HarmonicSweepStatus* status=nullptr;
    const HarmonicDeviceStatus* modes=nullptr; // two ordered passes, actual column stride
    MinimumNormDeviceView linear{};
};
struct HarmonicSweepResult {
    std::vector<Point> correction;
    std::vector<double> rhs;
    std::vector<HarmonicDeviceStatus> modes;
    HarmonicSweepStatus status;
};
// Streams harmonic modes into the Jacobian, solves the unchanged pivoted QR,
// and regenerates modes for ordered correction accumulation. Conditional graph
// loops, mode counts, coefficients and acceptance stay on device. No mode table
// of pointCount*columnCount vectors is stored.
class HarmonicSweepCorrection {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit HarmonicSweepCorrection(const HarmonicSweepDefinition&);
    ~HarmonicSweepCorrection();
    HarmonicSweepDeviceView view()const;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,
        const HarmonicSweepDeviceProblem* deviceProblem);
    void solveDevice(const HarmonicSweepDeviceProblem*,void* stream=nullptr);
    // Publication adapter for the remaining host Newton controller. Initial
    // geometry/modes/scales are retained from construction. This compatibility
    // call still forms an outer feedback loop; resident owners use appendTo.
    HarmonicSweepResult solveBorrowed(const Point* endpoint,const double* rhs,
        const double* consistencyTolerance,double rankTolerance);
    HarmonicSweepResult solve(const std::vector<Point>& endpoint,const std::vector<double>& rhs,
        double rankTolerance,double consistencyTolerance);
};
}}}
#endif
