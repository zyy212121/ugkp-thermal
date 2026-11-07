#ifndef UGKP_SWEEP_LINEARIZATION_RESIDENT_PROBE_H
#define UGKP_SWEEP_LINEARIZATION_RESIDENT_PROBE_H
#include "gpu/PolyhedralGeometry.H"
struct ResidentSweepProbeResult {
    std::vector<ugkp::gpu::geometry::Polynomial> polynomials;
    std::vector<std::vector<double>> columns;
    std::vector<ugkp::gpu::geometry::Point> correction;
};
ResidentSweepProbeResult residentSweepProbe(const ugkp::gpu::geometry::Topology&,
    const std::vector<ugkp::gpu::geometry::Point>& old,
    const std::vector<ugkp::gpu::geometry::Point>& endpoint,
    const std::vector<int>& faces,const std::vector<double>& scales,
    const std::vector<ugkp::gpu::geometry::PlanarConstraint>& planar,
    const std::vector<std::vector<ugkp::gpu::geometry::Point>>& modes,
    const std::vector<double>& coefficients);
#endif
