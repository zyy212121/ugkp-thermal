#pragma once

// to STL containers are placed in the gas state or checkpoint.
struct BoundaryLayerAllocation
{
    ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
    ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
    double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
};
struct SharedGasTrialSnapshot
{
    double* cells=nullptr;
    double* faces=nullptr;
    int* cellStatus=nullptr;
    int* faceStatus=nullptr;
    ugkwp::ChemistryStatus* chemistryStatus=nullptr;
    std::size_t cellScalars=0,faceScalars=0;
    unsigned char* wallRecords=nullptr;
    std::size_t wallRecordBytes=0;
    double wallStageTime=0,wallAuditStart=0,wallAuditInterval=0;
    bool valid=false;
};
struct SharedGasTrialStorage
{
    SharedGasTrialSnapshot trial;
    SharedGasTrialSnapshot interval;
    BoundaryLayerAllocation rejectedWall;
    double targetMaxCo=.5;
    bool retryableFailure=false;
    bool terminalContinuation=false;
};
