#pragma once
// One capacity/residency bound with native application residency policies.
inline void setParticleWorkGridFromResidency(DeviceState* s, const int residentBlocks)
{
    const int capacityGrid =
        (s->particleCapacity + s->particleBlockThreads - 1)/s->particleBlockThreads;
    const int saturationGrid = s->multiprocessorCount*residentBlocks;
    s->particleWorkGrid = capacityGrid < saturationGrid ? capacityGrid : saturationGrid;
}

#if !GPU_OPERATOR_THERMAL
int queryParticleKernelResidency(DeviceState* s, int& residentBlocks)
{
    int trackingBlocks = 0;
    int binningBlocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (&trackingBlocks, trackParticlesLocalFaceWalkKernel, s->particleBlockThreads, 0);
    if (err == cudaSuccess)
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (&binningBlocks, countParticlesByCellKernel<true>, s->particleBlockThreads, 0);
    if (err != cudaSuccess)
    {
        setLastError("particle launch occupancy query", err);
        return 1;
    }
    if (trackingBlocks <= 0 || binningBlocks <= 0)
    {
        setLastErrorText("selected particle block has zero kernel occupancy");
        return 1;
    }
    residentBlocks = trackingBlocks < binningBlocks ? trackingBlocks : binningBlocks;
    return 0;
}
#endif
