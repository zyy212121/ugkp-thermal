#ifndef UGKP_GPU_SWEEP_LINE_SEARCH_H
#define UGKP_GPU_SWEEP_LINE_SEARCH_H
#include "SweepResidual.H"
#include "ExecutionGraph.H"
namespace ugkp { namespace gpu { namespace geometry {
struct SweepLineSearchProblem {
    SweepResidualProblem residual;
    const Point* origin=nullptr;
    const Point* correction=nullptr;
    double initialMetric=0;
    int maximumTrials=1;
};
struct SweepLineSearchView {
    const Point* points=nullptr;
    const SweepLineSearchStatus* status=nullptr;
};
// Finds the next native Armijo candidate. Geometry certification may reject it
// and resume the same fixed-target search. The origin/correction must remain
// immutable until acceptance; this owner never commits a trial to either.
class ResidentSweepLineSearch {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentSweepLineSearch(int points,ResidentSweepResidual&);
    ~ResidentSweepLineSearch();
    SweepLineSearchView view()const;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,const SweepLineSearchProblem*,bool restart=true);
    void begin(const SweepLineSearchProblem*,void* stream=nullptr);
    void resume(void* stream=nullptr);
};
}}}
#endif
