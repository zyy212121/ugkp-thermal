#pragma once
// Particle work is bounded by capacity and the kernels it launches.
// Cell-reduction hierarchy and contact configuration do not select this grid.
inline void setParticleWorkGridFromResidency(DeviceState* s, const int residentBlocks)
{
    const int capacityGrid =
        s->particleCapacity/s->particleBlockThreads
        + (s->particleCapacity % s->particleBlockThreads != 0);
    const int saturationGrid = s->multiprocessorCount*residentBlocks;
    s->particleWorkGrid = capacityGrid < saturationGrid ? capacityGrid : saturationGrid;
}

template<class BinningKernel>
int queryParticleKernelResidency
(DeviceState* s, BinningKernel binningKernel, int& residentBlocks)
{
    int trackingBlocks = 0;
    int binningBlocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (&trackingBlocks, trackParticlesLocalFaceWalkKernel, s->particleBlockThreads, 0);
    if (err == cudaSuccess)
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (&binningBlocks, binningKernel, s->particleBlockThreads, 0);
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

// Tracking is a grid-stride kernel. Its occupancy depends on B2 and the
// compiled tracking kernel, not on the selected cell-reduction hierarchy.
inline int configureTrackingWorkGrid(DeviceState* s)
{
    int residentBlocks = 0;
    const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
    (
        &residentBlocks,
        trackParticlesLocalFaceWalkKernel,
        s->particleBlockThreads,
        0
    );
    if (err != cudaSuccess)
    {
        setLastError("tracking launch occupancy query", err);
        return 1;
    }
    if (residentBlocks <= 0)
    {
        setLastErrorText("selected particle block has zero tracking occupancy");
        return 1;
    }
    const int capacityGrid = s->particleCapacity/s->particleBlockThreads
        + (s->particleCapacity % s->particleBlockThreads != 0);
    const int saturationGrid = s->multiprocessorCount*residentBlocks;
    s->trackingWorkGrid = capacityGrid < saturationGrid
        ? capacityGrid : saturationGrid;
    std::fprintf(stderr,
        "Tracking geometry: B2=%d blocksPerSM=%d grid=%d\n",
        s->particleBlockThreads, residentBlocks, s->trackingWorkGrid);
    return 0;
}
