#ifndef UGKP_GPU_SWEEP_PLANARITY_H
#define UGKP_GPU_SWEEP_PLANARITY_H
#include "PolyhedralGeometry.H"
namespace ugkp { namespace gpu { namespace geometry {
struct SweepPlanarityInput {const Point* localPoints=nullptr;double coordinateUncertainty=0;};
struct SweepPlanarityView {
    const PlanarConstraint* planar=nullptr;
    const double* tolerance=nullptr;
    const int* count=nullptr;
};
// Candidate storage comes solely from topology. Current coordinate rounding
// determines the actual constraints; stable compaction preserves native face
// and vertex order, without padded equations or a host count round trip.
class ResidentSweepPlanarity {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit ResidentSweepPlanarity(const Topology&);
    ~ResidentSweepPlanarity();
    int capacity()const;
    SweepPlanarityView view()const;
    void evaluate(const SweepPlanarityInput*,void* stream=nullptr);
};
}}}
#endif
