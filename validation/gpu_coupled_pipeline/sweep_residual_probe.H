#pragma once
#include "gpu/SweepResidual.H"
struct SweepResidualProbe {
    ugkp::gpu::geometry::SweepResidualStatus status;
    std::vector<ugkp::gpu::geometry::SweepResult> sweeps;
    std::vector<double> residual,rhs;
};
SweepResidualProbe probeSweepResidual(const ugkp::gpu::geometry::SweepResidualDefinition&,
    const std::vector<ugkp::gpu::geometry::Point>&);

std::vector<double> probeSweepRatios(const std::vector<double>&,const std::vector<double>&);
