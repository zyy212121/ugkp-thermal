#pragma once
#include <cuda_runtime.h>
#include <cstddef>

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
    size_t enabledFaceCapacity = 0;
    // Host ownership until a replacement is published and the old list freed.
    int* retiredFaceIds = nullptr;
    double* particleWallDepositedEnergy = nullptr;
    double* particleWallReflectedEnergy = nullptr;
};

extern "C" int ugkpInvalidateGasWallEnergyGraph(void*);
extern "C" int ugkpPrepareParticleWallEnergyRestore(void*);

int ugkpAccumulateGasWallEnergy64(GpuWallEnergyState*, double, cudaStream_t);
void ugkpReleaseGasWallEnergy64(GpuWallEnergyState*);
