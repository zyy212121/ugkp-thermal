#pragma once
#include <cuda_runtime.h>

struct GpuWallEnergyState
{
    GpuWallEnergyState* deviceState = nullptr;
    char* errorBuffer = nullptr;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nEnabledFaces = 0;
    int particleWallHeatTransferEnabled = 0;
    double* gasWallEnergy = nullptr;
    unsigned char* gasWallEnergyMask = nullptr;
    double* gasWallFlux = nullptr;
    int* enabledFaceIds = nullptr;
    double* particleWallDepositedEnergy = nullptr;
    double* particleWallReflectedEnergy = nullptr;
};

int ugkpAccumulateGasWallEnergy64(GpuWallEnergyState*, double, cudaStream_t);
void ugkpReleaseGasWallEnergy64(GpuWallEnergyState*);
