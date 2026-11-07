#pragma once
#include "../../common/gpu/SweepPlanarity.H"
struct SweepPlanarityProbe {std::vector<ugkp::gpu::geometry::PlanarConstraint> planar;std::vector<double> tolerance;};
SweepPlanarityProbe probeSweepPlanarity(ugkp::gpu::geometry::ResidentSweepPlanarity&,const std::vector<ugkp::gpu::geometry::Point>&,double);
