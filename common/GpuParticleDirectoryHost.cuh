#pragma once

// Full and split-pre directory construction for the dynamic-heavy-policy path.
// Requires DeviceState, bin/split kernels, CUB, error helpers and
// updateDynamicHeavyPolicy/prepare{,Split}CsrHeavyReductionTasks.
// The caller retains ownership of scan storage and task scheduling policy.
// Gas's static-directory flag resets and producer protocol are a separate path.
int binParticlesByCell(DeviceState* s, const int block, const bool survivorsOnly = false)
{
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
    if (updateDynamicHeavyPolicy(s) != 0)
    {
        return 1;
    }
    return prepareCsrHeavyReductionTasks(s, block);
}

int buildSplitPreDirectory(DeviceState* s, const int block)
{
    if
    (
        s->csrCellLocalPathEnabled == 0
     || s->preBaseDirectoryReady == 0
    )
    {
        s->splitPreDirectoryActive = 0;
        return binParticlesByCell(s, block);
    }

    s->splitPreDirectoryActive = 1;

    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    clearSplitPreInjectionBinsKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clear split-Dpre injection bins launch", err);
        return 1;
    }

    if (s->nBoundarySources == 0)
    {
        if (updateDynamicHeavyPolicy(s) != 0)
        {
            return 1;
        }
        return prepareSplitCsrHeavyReductionTasks(s, block);
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

    initialiseSplitPreInjectionWritesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("initialise split-Dpre injection writes launch", err);
        return 1;
    }
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

    if (updateDynamicHeavyPolicy(s) != 0)
    {
        return 1;
    }
    return prepareSplitCsrHeavyReductionTasks(s, block);
}
