#pragma once
#include <cuda_runtime.h>
#include "../GpuPrecisionTypes.H"
#include <cstddef>
#include "../gasTransport/GasResidentInterval.H"
#include "../gpu/RadiationResidentView.H"
#include "../gpu/ThermalInterfaceDeviceView.H"

struct GpuWallEnergyState
{
    GpuWallEnergyState* deviceState = nullptr;
    char* errorBuffer = nullptr;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nEnabledFaces = 0;
    int particleWallHeatTransferEnabled = 0;
    GpuWallEnergy* gasWallEnergy = nullptr;
    unsigned char* gasWallEnergyMask = nullptr;
    GpuWallEnergy* gasWallFlux = nullptr;
    int* enabledFaceIds = nullptr;
    size_t enabledFaceCapacity = 0;

    // Persistent compact fluid/solid interface map and exchange workspace.
    int nInterfaceSolidFaces = 0;
    int* interfaceRowOffsets = nullptr;
    int* interfaceSourceIndices = nullptr;
    GpuReal* interfaceAreas = nullptr;
    GpuWallEnergy* interfaceRadiationEnergy = nullptr;
    GpuWallEnergy* interfaceCombinedEnergy = nullptr;
    GpuWallEnergy* interfacePreview = nullptr;
    double* interfaceInterval = nullptr;
    GpuReal* interfaceEmissivityByFace = nullptr;
    cudaEvent_t interfaceInputReady = nullptr;

    int* retiredFaceIds = nullptr;
    GpuWallEnergy* particleWallDepositedEnergy = nullptr;
    GpuWallEnergy* particleWallReflectedEnergy = nullptr;
};

extern "C" int ugkpInvalidateGasWallEnergyGraph(void*);
extern "C" int ugkpPrepareParticleWallEnergyRestore(void*);

int ugkpAccumulateGasWallEnergy64(GpuWallEnergyState*, double, cudaStream_t,
    const ugkwp::GasResidentInterval* interval=nullptr);
void ugkpReleaseGasWallEnergy64(GpuWallEnergyState*);
