#include "GasBuildConfig.H"
#pragma once


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
