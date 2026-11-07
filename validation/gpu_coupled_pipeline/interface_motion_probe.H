#pragma once
#include "gpu/InterfaceGasPointMotion.H"
ugkp::gpu::geometry::InterfaceGasPointMotionResult probeInterfaceMotion(
    ugkp::gpu::geometry::InterfaceGasPointMotion&,const ugkp::gpu::geometry::InterfaceGasPointMotionDefinition&,
    const std::vector<ugkp::gpu::geometry::Point>&,const std::vector<ugkp::gpu::geometry::Point>&);
