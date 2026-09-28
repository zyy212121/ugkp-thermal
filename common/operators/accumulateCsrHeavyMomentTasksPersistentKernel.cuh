#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void accumulateCsrHeavyMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int task;
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (task >= *s.csrHeavyTaskCount)
        {
            return;
        }

        const int c = s.csrHeavyTaskCell[task];
        GPU_OPERATOR_REAL sums[8];
        accumulateCsrHeavyMomentTask
        (
            s,
            c,
            s.csrHeavyTaskBegin[task],
            s.csrHeavyTaskEnd[task],
            sums,
            warpPartials
        );
        if (threadIdx.x == 0)
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
        __syncthreads();
    }
}

__global__ void finalizeCsrHeavyMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int heavyCellIndex;
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            heavyCellIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (heavyCellIndex >= *s.csrHeavyCellCount)
        {
            return;
        }

        const int c = s.csrHeavyCellList[heavyCellIndex];
        GPU_OPERATOR_REAL sums[8] = {GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0)};
        const int firstTask = s.csrHeavyCellTaskStart[c];
        const int nTasks = s.csrHeavyCellTaskCount[c];
        for
        (
            int localTask = threadIdx.x;
            localTask < nTasks;
            localTask += blockDim.x
        )
        {
            const int task = firstTask + localTask;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] +=
                    s.csrHeavyPartials
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
            if (c == 0)
            {
                s.cellParticleCount[s.nCells] = 0;
            }
            const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], s.rhoMin);
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
