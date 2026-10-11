#pragma once
// Device allocation ownership for the two transaction levels. No host pointers
// to STL containers are placed in the gas state or checkpoint.
struct SharedGasTrialSnapshot
{
    double* cells=nullptr;
    double* faces=nullptr;
    int* cellStatus=nullptr;
    int* faceStatus=nullptr;
    ugkwp::ChemistryStatus* chemistryStatus=nullptr;
    ugkwp::ChemistryAudit<double,ugkwp::compiledGasSpecies>* chemistryAudits=nullptr;
    unsigned char* wallRecords=nullptr;
    std::size_t wallRecordBytes=0;
    double wallStageTime=0,wallAuditStart=0,wallAuditInterval=0;
    bool valid=false;
};
struct SharedGasTrialStorage
{
    SharedGasTrialSnapshot trial;
    SharedGasTrialSnapshot interval;
    ugkwp::ChemistryAudit<double,ugkwp::compiledGasSpecies>* chemistryBefore=nullptr;
    ugkwp::ChemistryAudit<double,ugkwp::compiledGasSpecies>* chemistryAfter=nullptr;
    double targetMaxCo=0.5;
    bool retryableFailure=false;
};
