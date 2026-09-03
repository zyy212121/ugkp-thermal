#pragma once

template<bool PoissonMode>
__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
    __shared__ double collisionProbability;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            taskCount = *s.csrHeavyTaskCount;
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            if (task < taskCount)
            {
                descriptor = s.csrReductionTasks[task];
                if (PoissonMode)
                {
                    const double tauColl =
                        granularCollisionTauFromCellDevice(s, descriptor.cell);
                    collisionProbability =
                        (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
                      ? 0.0
                      : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
                }
                else collisionProbability = 1.0;
            }
        }
        __syncthreads();
        if (task >= taskCount) return;
        const int c = descriptor.cell;
            if (PoissonMode && collisionProbability <= 0.0)
            {
                if (threadIdx.x == 0 && s.csrCellTaskCount[c] > 1)
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = 0.0;
                    }
                }
                __syncthreads();
                continue;
            }
            double sums[8];
            if
            (
                descriptor.source == static_cast<int>
                (
                    CsrReductionTaskSource::splitLogical
                )
            )
            {
                accumulateCsrSplitLogicalPoolTask<PoissonMode>
                (
                    s, c, descriptor.begin, descriptor.end,
                    collisionProbability, sums, warpPartials
                );
            }
            else
            {
                const bool directParticleIndex =
                    descriptor.source == static_cast<int>
                    (
                        CsrReductionTaskSource::splitBaseDirect
                    );
                accumulateCsrHeavyPoolTask<PoissonMode>
                (
                    s, c, descriptor.begin, descriptor.end,
                    directParticleIndex, collisionProbability,
                    sums, warpPartials
                );
            }
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.poissonPoolMass[c] = sums[0];
                    s.poissonPoolMomX[c] = sums[1];
                    s.poissonPoolMomY[c] = sums[2];
                    s.poissonPoolMomZ[c] = sums[3];
                    s.poissonPoolEnergy[c] = sums[4];
                    s.poissonPoolDiameter[c] = sums[5];
                    s.poissonPoolDiameter2[c] = sums[6];
                    s.poolThermalCount[c] = static_cast<int>(sums[7]);
                }
                else
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = sums[component];
                    }
                }
            }
            __syncthreads();
    }
}

__global__ void finalizeCsrSegmentedPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int multiIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0) multiIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        __syncthreads();
        if (multiIndex >= *s.csrHeavyCellCount) return;
        const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1)) asm("trap;");
        double sums[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = s.csrCellTaskOffset[c + 1];
        for (int task = firstTask + threadIdx.x; task < endTask; task += blockDim.x)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ];
            }
        }
        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.poissonPoolMass[c] = sums[0];
            s.poissonPoolMomX[c] = sums[1];
            s.poissonPoolMomY[c] = sums[2];
            s.poissonPoolMomZ[c] = sums[3];
            s.poissonPoolEnergy[c] = sums[4];
            s.poissonPoolDiameter[c] = sums[5];
            s.poissonPoolDiameter2[c] = sums[6];
            s.poolThermalCount[c] = static_cast<int>(sums[7]);
        }
        __syncthreads();
    }
}

int launchCsrSegmentedPoolReduction
(
    DeviceState* s,
    const double dt,
    const bool poissonMode,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes = 8u*static_cast<size_t>(warpCount)*sizeof(double);
    if (poissonMode)
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented pool worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells ? s->multiprocessorCount : s->nCells;
    finalizeCsrSegmentedPoolCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrSegmentedPoolCellsKernel launch", err);
        return 1;
    }
    return 0;
}
