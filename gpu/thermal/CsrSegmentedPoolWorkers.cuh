#include "GpuPrecisionTypes.H"
#pragma once
#include "CsrPersistentQueue.cuh"

template<bool PoissonMode>
__device__ __forceinline__ void executeCsrSegmentedPoolTask
(
    DeviceState& s, const int task, const CsrReductionTask& descriptor,
    const GpuReal collisionProbability, GpuReal* warpPartials
)
{
    const int c = descriptor.cell;
    if (PoissonMode && collisionProbability <= GPU_R(0.0))
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
                ] = GPU_R(0.0);
            }
        }
        return;
    }
    GpuReal sums[8];
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
}

template<bool PoissonMode>
struct ThermalPoolOperation
{
    CsrReductionTask* descriptor;
    GpuReal* probability;
    GpuReal* warpPartials;
    GpuTime dt;
    __device__ bool prepare(DeviceState& s, const int task)
    {
        *descriptor = s.csrReductionTasks[task];
        if constexpr (PoissonMode)
        {
            const GpuReal tau = granularCollisionTauFromCellDevice(s, descriptor->cell);
            *probability = (!(tau < GPU_R(0.5)*OfGreat) || tau <= OfSmall)
              ? GPU_R(0.0) : clampRange(GPU_R(1.0)-exp(-dt/tau), GPU_R(0.0), GPU_R(1.0));
        }
        else *probability = GPU_R(1.0);
        return true;
    }
    __device__ void execute(DeviceState& s, const int task)
    {
        executeCsrSegmentedPoolTask<PoissonMode>
            (s, task, *descriptor, *probability, warpPartials);
    }
};

template<bool PoissonMode>
__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel
(DeviceState* sp, const GpuTime dt)
{
    DeviceState& s = *sp;
    __shared__ CsrReductionTask descriptor;
    __shared__ GpuReal probability;
    extern __shared__ GpuReal warpPartials[];
    ThermalPoolOperation<PoissonMode> operation{&descriptor, &probability, warpPartials, dt};
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}

cudaError_t thermalPoolLaunchOccupancy(int* count, int block, size_t sharedBytes)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (count, accumulateCsrSegmentedPoolTasksPersistentKernel<true>, block, sharedBytes);
}

struct ThermalPoolFinalizeOperation
{
    GpuReal* warpPartials;
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int multiIndex)
    {
    const int c = s.csrMultiTaskCellList[multiIndex];
    if (!(s.csrCellTaskCount[c] > 1)) asm("trap;");
    GpuReal sums[8] = {GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
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

    }
};

__global__ void finalizeCsrSegmentedPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    extern __shared__ GpuReal warpPartials[];
    ThermalPoolFinalizeOperation operation{warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyCellCount, operation);
}

int launchCsrSegmentedPoolReduction
(
    DeviceState* s,
    const GpuTime dt,
    const bool poissonMode,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err;
    err = resetCsrPersistentQueue(s);
    if (err != cudaSuccess)
    {
        setLastError("reset CSR persistent queue", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes = 8u*static_cast<size_t>(warpCount)*sizeof(GpuReal);
    if (poissonMode)
        accumulateCsrSegmentedPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    else
        accumulateCsrSegmentedPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented pool worker launch", err);
        return 1;
    }
    const int finalizeGrid =
        std::min(s->multiprocessorCount, s->nCells);
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
