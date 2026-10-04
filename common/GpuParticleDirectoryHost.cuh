#pragma once

// One host protocol for full/split directory preparation and task readiness.
#include "GpuParticleDirectoryHostPolicy.cuh"

int binParticlesByCell(DeviceState* s, const int block, const bool survivorsOnly = false)
{
    selectParticleDirectory(s, false);
    s->csrTasksReady = 0;
    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    clearParticleCellBinsKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clearParticleCellBinsKernel launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        if (survivorsOnly)
            countParticlesByCellKernel<true, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            countParticlesByCellKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        if (survivorsOnly)
            countParticlesByCellKernel<false, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            countParticlesByCellKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("countParticlesByCellKernel launch", err);
        return 1;
    }

    err = cub::DeviceScan::ExclusiveSum
    (
        s->cellScanTempStorage,
        s->cellScanTempBytes,
        s->cellParticleCount,
        s->cellParticleOffset,
        s->nCells + 1
    );
    if (err != cudaSuccess)
    {
        setLastError("cell particle count exclusive scan", err);
        return 1;
    }

    initialiseParticleCellWritesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("initialiseParticleCellWritesKernel launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        if (survivorsOnly)
            scatterParticlesByCellKernel<true, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            scatterParticlesByCellKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        if (survivorsOnly)
            scatterParticlesByCellKernel<false, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            scatterParticlesByCellKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatterParticlesByCellKernel launch", err);
        return 1;
    }
    return prepareParticleDirectoryTasks(s, block, fullParticleDirectoryKind());
}

#if GPU_DIRECTORY_HAS_BASE_ONLY
int prepareSourceFreeSplitPreDirectory(DeviceState* s)
{
    cudaError_t err = cudaMemset
    (
        s->cellParticleOffset,
        0,
        static_cast<size_t>(s->nCells + 1)*sizeof(int)
    );
    if (err != cudaSuccess)
    {
        setLastError("reset source-free split-Dpre injection offsets", err);
        return 1;
    }

    s->preInjectionSegmentActive = 0;
    s->useSplitPreDirectory = 1;
    return 0;
}
#endif

int preparePreTransportParticleDirectory(DeviceState* s, const int block)
{
    if
    (
        !splitParticleDirectoryEnabled(s)
     || s->preBaseDirectoryReady == 0
    )
    {
        return binParticlesByCell(s, block);
    }

#if GPU_DIRECTORY_HAS_BASE_ONLY
    // Compaction already built the direct-base tasks. No injection means the
    // next step can retain their contents and publication tag unchanged.
    if (s->nBoundarySources == 0) return prepareSourceFreeSplitPreDirectory(s);
#endif
    selectParticleDirectory(s, true);
    s->csrTasksReady = 0;

    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    if (clearParticleInjectionBins(s, cellGrid, block) != 0) return 1;
    cudaError_t err = cudaSuccess;

    if (s->nBoundarySources == 0)
    {
        return prepareParticleDirectoryTasks(s, block, splitParticleDirectoryKind());
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        countSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        countSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("count split-Dpre injection particles launch", err);
        return 1;
    }

    err = cub::DeviceScan::ExclusiveSum
    (
        s->cellScanTempStorage,
        s->cellScanTempBytes,
        s->cellParticleCount,
        s->cellParticleOffset,
        s->nCells + 1
    );
    if (err != cudaSuccess)
    {
        setLastError("split-Dpre injection count exclusive scan", err);
        return 1;
    }

    if (initialiseParticleInjectionWrites(s, cellGrid, block) != 0) return 1;
    if (s->csrWarpAggregatedBinning != 0)
    {
        scatterSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        scatterSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatter split-Dpre injection particles launch", err);
        return 1;
    }

    return prepareParticleDirectoryTasks(s, block, splitParticleDirectoryKind());
}

// Preparation, directory classification and automatic task readiness are one
// operation, so production callers cannot pass a mismatched directory kind.
int prepareParticleDirectoryAndSchedule(DeviceState* s, const int block)
{
    if (s->csrCellLocalPathEnabled == 0) return 0;
    if (preparePreTransportParticleDirectory(s, block) != 0) return 1;
    return runAutomaticCsrSchedule(s, block, currentPreTransportDirectoryKind(s));
}

int buildSplitPreDirectory(DeviceState* s, const int block)
{
    return preparePreTransportParticleDirectory(s, block);
}

int buildPostTransportDirectory(DeviceState* s, const int block)
{
    return binParticlesByCell(s, block, true);
}
