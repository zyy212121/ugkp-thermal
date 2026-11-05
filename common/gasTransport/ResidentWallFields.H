#include "GasBuildConfig.H"
#pragma once

// to STL containers are placed in the gas state or checkpoint.
struct BoundaryLayerAllocation
{
    ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
    ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
    double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
};
struct SharedGasTrialStorage
{
    BoundaryLayerAllocation rejectedWall;
};

inline int releaseSharedGasTrialStorage(SharedGasTrialStorage&){return 0;}
