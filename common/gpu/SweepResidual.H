#ifndef UGKP_GPU_SWEEP_RESIDUAL_H
#define UGKP_GPU_SWEEP_RESIDUAL_H
#include "PolyhedralGeometry.H"
#include "SweepResidualMath.H"
namespace ugkp { namespace gpu { namespace geometry {
enum class SweepResidualFailure:int {None,Endpoint,EmptyNormal,EmptyTangential,Sweep,FaceResidual,PlanarResidual,Metric};
struct SweepResidualStatus {
    double metric=0,maximumSweep=0,maximumPlanarity=0,consistencyTolerance=0;
    SweepResidualFailure failure=SweepResidualFailure::None;
    int index=-1;
};
struct SweepResidualProblem {
    const Point* old=nullptr;const Point* endpoint=nullptr;
    const double *targets=nullptr,*scales=nullptr,*faceCeiling=nullptr,*planeTolerance=nullptr;
    const PlanarConstraint* planar=nullptr;
    double absoluteTolerance=0,relativeTolerance=0,rankTolerance=0;
};
struct SweepResidualView {
    const SweepResult* sweptFaces=nullptr;
    const double* residual=nullptr;
    const double* rhs=nullptr;
    const SweepResidualStatus* status=nullptr;
};
// Topology/Empty pairing are validated once by the mesh owner. Coordinates,
// targets, scales and controls are borrowed from their resident producers.
class ResidentSweepResidual {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentSweepResidual(const Topology&,int points,const std::vector<int>& faces,
        int planes,int emptyAxis=-1,const std::vector<int>& emptyMate={});
    ~ResidentSweepResidual();
    SweepResidualView view()const;
    void evaluate(const SweepResidualProblem*,void* stream=nullptr);
};
struct SweepResidualDefinition {
    Topology topology;std::vector<Point> old;
    std::vector<int> faces,emptyMate;
    std::vector<double> targets,scales,faceCeiling,planeTolerance;
    std::vector<PlanarConstraint> planar;
    double absoluteTolerance=0,relativeTolerance=0,rankTolerance=0;
    int emptyAxis=-1;
};
// Compatibility publication for the remaining host line search. Residual/RHS
// and endpoint also stay available to the next resident Newton operator.
class SweepResidualEvaluation {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit SweepResidualEvaluation(const SweepResidualDefinition&);
    ~SweepResidualEvaluation();
    SweepResidualView view()const;
    const Point* endpoint()const;
    SweepResidualStatus evaluate(const std::vector<Point>&);
};
}}}
#endif
