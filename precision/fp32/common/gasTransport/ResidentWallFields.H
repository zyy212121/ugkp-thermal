#include "../GpuPrecisionTypes.H"
#include "GasBuildConfig.H"
#pragma once

// to STL containers are placed in the gas state or checkpoint.
struct BoundaryLayerAllocation
{
    ugkwp::GasBoundaryLayerState<GpuReal> gasBoundaryLayer;
    ugkwp::GasBoundaryLayerModelState<GpuReal,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
    GpuReal *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
};
struct SharedGasTrialStorage
{
    BoundaryLayerAllocation rejectedWall;
};

inline int releaseSharedGasTrialStorage(SharedGasTrialStorage&){return 0;}
