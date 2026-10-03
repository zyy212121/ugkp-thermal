#pragma once
// Shared S2 task execution and reduction. Physical accumulation/copy hooks are
// provided by GpuParticleMoments.cuh; application identity is not a policy.
#include "CsrPersistentQueue.cuh"
#include "GpuLaunchOptions.cuh"
template<bool GatherSurvivors = false>
struct CsrMomentOperation
{
    CsrReductionTask* current;
    GPU_PIPELINE_REAL* warpPartials;
    __device__ bool prepare(DeviceState& s, const int task)
    {
        *current = s.csrReductionTasks[task];
        return true;
    }
    __device__ void execute(DeviceState& s, const int task)
    {
        const CsrReductionTask& descriptor = *current;
        const int c = descriptor.cell;
            GPU_PIPELINE_REAL sums[8];
            accumulateCsrHeavyMomentTask<GatherSurvivors>
            (
                s, c, descriptor.begin, descriptor.end, sums, warpPartials
            );
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    publishParticleMomentsCell(s, c, sums);
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
};

template<bool GatherSurvivors = false>
__global__ void accumulateCsrSegmentedMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ CsrReductionTask descriptor;
    extern __shared__ GPU_PIPELINE_REAL warpPartials[];
    CsrMomentOperation<GatherSurvivors> operation{&descriptor, warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}

__device__ __forceinline__ void finalizeCsrMomentCell
(DeviceState& s,const int c,GPU_PIPELINE_REAL* warpPartials)
{
        if (!(s.csrCellTaskCount[c] > 1))
        {
            asm("trap;");
        }
        GPU_PIPELINE_REAL sums[8] = {GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0), GPU_PIPELINE_REAL(0.0)};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = firstTask + s.csrCellTaskCount[c];
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
            publishParticleMomentsCell(s, c, sums);
        }

}

struct CsrMomentFinalizeOperation
{
    GPU_PIPELINE_REAL* warpPartials;
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int multiIndex)
    {
        finalizeCsrMomentCell(s,s.csrMultiTaskCellList[multiIndex],warpPartials);
    }

};

__global__ void finalizeCsrSegmentedMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    extern __shared__ GPU_PIPELINE_REAL warpPartials[];
    CsrMomentFinalizeOperation operation{warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyCellCount, operation);
}

int launchCommonSegmentedMomentReduction
(DeviceState* s, const int block, const SegmentedMomentOptions options)
{
    const bool gatherSurvivors = options.payload == MomentPayload::gatherSurvivors;
    const bool deferRecovery = options.recovery == MomentRecovery::deferToAdvance;
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
        8u*static_cast<size_t>(warpCount)*sizeof(GPU_PIPELINE_REAL);
    if (gatherSurvivors)
        accumulateCsrSegmentedMomentTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    else
        accumulateCsrSegmentedMomentTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented moment worker launch", err);
        return 1;
    }
    // Only advance has a following fused recovery; restart must complete here.
    if (!deferRecovery)
    {
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
        finalizeCsrSegmentedMomentCellsKernel
            <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("finalizeCsrSegmentedMomentCellsKernel launch", err);
            return 1;
        }
    }
    return 0;
}

#undef GPU_PIPELINE_REAL
