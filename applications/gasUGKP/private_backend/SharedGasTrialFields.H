#pragma once
// Device allocation ownership for genuine failed-operation restoration. No host pointers
// to STL containers are placed in the gas state or checkpoint.
struct BoundaryLayerAllocation
{
    ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
    ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
    ugkwp::GasSstAuditState<double> gasSstAudit;
    double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
};
struct SharedGasTrialSnapshot
{
    double* cells=nullptr;
    double* faces=nullptr;
    int* cellStatus=nullptr;
    int* faceStatus=nullptr;
    ugkwp::ChemistryStatus* chemistryStatus=nullptr;
    ugkwp::ChemistryAudit<double,ugkwp::compiledGasSpecies>* chemistryAudit=nullptr;
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
