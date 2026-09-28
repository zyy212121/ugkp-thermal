#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void finalizeCsrHeavyPoolCellsKernel(DeviceState* sp)
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
        GPU_OPERATOR_REAL sums[8] =
        {
            GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0)
        };
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
