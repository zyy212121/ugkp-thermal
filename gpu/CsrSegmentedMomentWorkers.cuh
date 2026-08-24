#pragma once

__global__ void accumulateCsrSegmentedMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
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
            }
        }
        __syncthreads();
        if (task >= taskCount) return;
        const int c = descriptor.cell;
            double sums[8];
            accumulateCsrHeavyMomentTask
            (
                s, c, descriptor.begin, descriptor.end, sums, warpPartials
            );
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.cellParticleCount[c] = static_cast<int>(sums[7]);
                    if (c == 0) s.cellParticleCount[s.nCells] = 0;
                    const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
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
}

__global__ void finalizeCsrSegmentedMomentCellsKernel(DeviceState* sp)
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
            s.cellParticleCount[c] = static_cast<int>(sums[7]);
            if (c == 0) s.cellParticleCount[s.nCells] = 0;
            const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
            s.momRhoP[c] = sums[0]*invV;
            s.momRhoUPx[c] = sums[1]*invV;
            s.momRhoUPy[c] = sums[2]*invV;
            s.momRhoUPz[c] = sums[3]*invV;
            s.momRhoEP[c] = sums[4]*invV;
            s.momRhoPD[c] = sums[5]*invV;
            s.momRhoHpP[c] = sums[6]*invV;
        }
        __syncthreads();
    }
}

int launchCsrSegmentedMomentReduction(DeviceState* s, const int block)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes = 8u*static_cast<size_t>(warpCount)*sizeof(double);
    accumulateCsrSegmentedMomentTasksPersistentKernel
        <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented moment worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells ? s->multiprocessorCount : s->nCells;
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
