#pragma once
#include "GpuPoolMomentOperations.cuh"
#include "CsrPersistentQueue.cuh"
// Common pool queue operation, partial publication, finalization and launch protocol.
template<bool PoissonMode
#if GPU_POOL_STATIC_DIRECTORY
    , HeavyDirectoryKind DirectoryKind = HeavyDirectoryKind::full
#endif
>
struct CsrPoolOperation
{
    CsrReductionTask* current;
    GPU_OPERATOR_REAL* probability;
    GPU_OPERATOR_REAL* warpPartials;
    GPU_OPERATOR_TIME dt;
    __device__ bool prepare(DeviceState& s, const int task)
    {
        *current = s.csrReductionTasks[task];
        if constexpr (PoissonMode)
        {
#if GPU_POOL_CACHED_PROBABILITY
            *probability = s.csrCellTaskCount[current->cell] > 1
                ? s.poissonCellCollisionProbability[current->cell]
                : poissonCollisionProbabilityForCell(s, current->cell, dt);
#else
            *probability = poissonCollisionProbabilityForCell(s, current->cell, dt);
#endif
        }
        else *probability = GPU_OPERATOR_R(1.0);
        return true;
    }
    __device__ void execute(DeviceState& s, const int task)
    {
        const CsrReductionTask& descriptor = *current;
        const GPU_OPERATOR_REAL collisionProbability = *probability;
        const int c = descriptor.cell;
            if (PoissonMode && collisionProbability <= GPU_OPERATOR_R(0.0))
            {
                if (threadIdx.x == 0 && s.csrCellTaskCount[c] > 1)
                {
                    zeroPoolPartial(s, task);
                }
                return;
            }
            GPU_OPERATOR_REAL sums[8];
            if
            (
                GPU_POOL_SPLIT_DIRECTORY
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
                    GPU_POOL_DIRECT_PARTICLE_INDEX;
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
                    publishPoolCell<false>(s, c, sums);
                    GPU_POOL_SAMPLING_READY(s, c)
                }
                else
                {
                    publishPoolPartial(s, task, sums);
                }
            }

    }
};

template<bool PoissonMode
#if GPU_POOL_STATIC_DIRECTORY
    , HeavyDirectoryKind DirectoryKind = HeavyDirectoryKind::full
#endif
>
__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel
(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    __shared__ CsrReductionTask descriptor;
    __shared__ GPU_OPERATOR_REAL probability;
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];
    CsrPoolOperation<PoissonMode
#if GPU_POOL_STATIC_DIRECTORY
        , DirectoryKind
#endif
    > operation
        {&descriptor, &probability, warpPartials, dt};
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}

struct CsrPoolFinalizeOperation
{
    GPU_OPERATOR_REAL* warpPartials;
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int multiIndex)
    {
    const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1))
        {
            asm("trap;");
        }
        GPU_OPERATOR_REAL sums[8] = {GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0)};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = GPU_POOL_END_TASK(s, c, firstTask);
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
        reducePoolMoments<PoolReductionTopology::warpComponents>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            publishPoolCell<false>(s, c, sums);
                    GPU_POOL_SAMPLING_READY(s, c)
        }

    }
};

__global__ void finalizeCsrSegmentedPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];
    CsrPoolFinalizeOperation operation{warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyCellCount, operation);
}

int launchCsrSegmentedPoolReduction
(
    DeviceState* s,
    const GPU_OPERATOR_TIME dt,
    const bool poissonMode,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    cudaError_t err = cudaSuccess;
    err = resetCsrPersistentQueue(s);
    if (err != cudaSuccess)
    {
        setLastError("reset CSR persistent queue", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(GPU_OPERATOR_REAL);
    if (poissonMode)
    {
#if GPU_POOL_STATIC_DIRECTORY
        if (s->csrReductionDirectoryKind == static_cast<int>(HeavyDirectoryKind::full))
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::full>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
        else if (s->csrReductionDirectoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::baseOnly>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
        else
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::splitBaseAndInjection>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
#else
        accumulateCsrSegmentedPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
#endif
    }
    else
    {
#if GPU_POOL_STATIC_DIRECTORY
        if (s->csrReductionDirectoryKind == static_cast<int>(HeavyDirectoryKind::full))
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<false, HeavyDirectoryKind::full>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
        else if (s->csrReductionDirectoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<false, HeavyDirectoryKind::baseOnly>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
        else
        {
            accumulateCsrSegmentedPoolTasksPersistentKernel<false, HeavyDirectoryKind::splitBaseAndInjection>
                <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        }
#else
        accumulateCsrSegmentedPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
#endif
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented pool worker launch", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells
      ? s->multiprocessorCount
      : s->nCells;
    err = resetCsrPersistentQueue(s);
    if (err != cudaSuccess)
    {
        setLastError("reset CSR persistent queue", err);
        return 1;
    }
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


#if !GPU_POOL_STATIC_DIRECTORY
cudaError_t thermalPoolLaunchOccupancy(int* count, int block, size_t sharedBytes)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (count, accumulateCsrSegmentedPoolTasksPersistentKernel<true>, block, sharedBytes);
}
#endif
