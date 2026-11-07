#ifndef UGKP_GPU_SWEEP_TRIAL_H
#define UGKP_GPU_SWEEP_TRIAL_H
#include "PolyhedralGeometry.H"
#include "SweepTrialMath.H"
namespace ugkp { namespace gpu { namespace geometry {
struct SweepTrialProblem {
    const Point* origin=nullptr;
    const Point* correction=nullptr;
    int backtrack=0;
};
struct SweepTrialStatus {
    double alpha=0;
    bool representable=true,changed=false;
    int firstInvalidPoint=-1;
};
struct SweepTrialView {
    const Point* points=nullptr;
    const SweepTrialStatus* status=nullptr;
};
// Coordinate generation and its stopping predicates are resident. The caller
// supplies the current Newton state and correction; no trial is committed here.
class ResidentSweepTrial {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit ResidentSweepTrial(int points);
    ~ResidentSweepTrial();
    SweepTrialView view()const;
    void evaluate(const SweepTrialProblem*,void* stream=nullptr);
};
}}}
#endif
