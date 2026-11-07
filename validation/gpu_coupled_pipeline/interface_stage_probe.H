#pragma once
#include "../../common/gpu/InterfaceGasStage.H"
ugkp::gpu::geometry::InterfaceGasStageResult probeInterfaceStage(ugkp::gpu::geometry::InterfaceGasStage&,
    const ugkp::gpu::geometry::InterfaceGasStageDefinition&,const std::vector<ugkp::gpu::geometry::Point>&,
    const std::vector<ugkp::gpu::geometry::Point>&,ugkp::gpu::geometry::StageGeometryControls);
