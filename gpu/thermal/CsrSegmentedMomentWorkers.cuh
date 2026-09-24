#include "GpuPrecisionTypes.H"
#pragma once
#include "CsrPersistentQueue.cuh"

template<bool GatherSurvivors = false>
struct ThermalMomentOperation
{
    CsrReductionTask* current;
    GpuReal* warpPartials;
    __device__ bool prepare(DeviceState& s, const int task)
    { *current = s.csrReductionTasks[task]; return true; }
    __device__ void execute(DeviceState& s, const int task)
    {
        const CsrReductionTask& descriptor = *current;
    const int c = descriptor.cell;
    GpuReal sums[8];
    accumulateCsrHeavyMomentTask<GatherSurvivors>
    (
        s, c, descriptor.begin, descriptor.end, sums, warpPartials
    );
    if (threadIdx.x == 0)
    {
        if (s.csrCellTaskCount[c] == 1)
        {
            s.cellParticleCount[c] = static_cast<int>(sums[7]);
            if (c == 0) s.cellParticleCount[s.nCells] = 0;
            const GpuReal invV = GPU_R(1.0)/clampMin(s.V[c], s.rhoMin);
            s.momRhoP[c] = sums[0]*invV;
            s.momRhoUPx[c] = sums[1]*invV;
            s.momRhoUPy[c] = sums[2]*invV;
            s.momRhoUPz[c] = sums[3]*invV;
            s.momRhoEP[c] = sums[4]*invV;
            s.momRhoPD[c] = sums[5]*invV;
            s.momRhoHpP[c] = sums[6]*invV;
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
    extern __shared__ GpuReal warpPartials[];
    ThermalMomentOperation<GatherSurvivors> operation{&descriptor, warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}

struct ThermalMomentFinalizeOperation
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
        s.cellParticleCount[c] = static_cast<int>(sums[7]);
        if (c == 0) s.cellParticleCount[s.nCells] = 0;
        const GpuReal invV = GPU_R(1.0)/clampMin(s.V[c], s.rhoMin);
        s.momRhoP[c] = sums[0]*invV;
        s.momRhoUPx[c] = sums[1]*invV;
        s.momRhoUPy[c] = sums[2]*invV;
        s.momRhoUPz[c] = sums[3]*invV;
        s.momRhoEP[c] = sums[4]*invV;
        s.momRhoPD[c] = sums[5]*invV;
        s.momRhoHpP[c] = sums[6]*invV;
    }

    }
};

__global__ void finalizeCsrSegmentedMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    extern __shared__ GpuReal warpPartials[];
    ThermalMomentFinalizeOperation operation{warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyCellCount, operation);
}

int launchCsrSegmentedMomentReduction(DeviceState* s, const int block, const bool gatherSurvivors = false)
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
    const int finalizeGrid =
        std::min(s->multiprocessorCount, s->nCells);
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
    return 0;
}
