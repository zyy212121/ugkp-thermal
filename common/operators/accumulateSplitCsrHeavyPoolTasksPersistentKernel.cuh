#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool DirectBase>
__global__ void accumulateSplitCsrHeavyPoolTasksPersistentKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ GPU_OPERATOR_REAL collisionProbability;
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];
    int* const cursor = DirectBase
      ? s.csrHeavyTaskCursor
      : s.csrHeavyInjectionTaskCursor;
    const int* const count = DirectBase
      ? s.csrHeavyTaskCount
      : s.csrHeavyInjectionTaskCount;

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(cursor, 1);
        }
        __syncthreads();
        if (task >= *count)
        {
            return;
        }

        const int c = DirectBase
          ? s.csrHeavyTaskCell[task]
          : s.csrHeavyInjectionTaskCell[task];
        if (threadIdx.x == 0)
        {
            const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);
            collisionProbability =
                (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
              ? GPU_OPERATOR_R(0.0)
              : clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        }
        __syncthreads();

        GPU_OPERATOR_REAL sums[8];
        const int begin = DirectBase
          ? s.csrHeavyTaskBegin[task]
          : s.csrHeavyInjectionTaskBegin[task];
        const int end = DirectBase
          ? s.csrHeavyTaskEnd[task]
          : s.csrHeavyInjectionTaskEnd[task];
        accumulateCsrHeavyPoolTask<true>
        (
            s,
            c,
            begin,
            end,
            DirectBase,
            collisionProbability,
            sums,
            warpPartials
        );
        if (threadIdx.x == 0)
        {
            GPU_OPERATOR_REAL* const partials = DirectBase
              ? s.csrHeavyPartials
              : s.csrHeavyInjectionPartials;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                partials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ] = sums[component];
            }
        }
        __syncthreads();
    }
}

__global__ void finalizeSplitCsrHeavyPoolCellsKernel(DeviceState* sp)
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
        const int baseFirst = s.csrHeavyCellTaskStart[c];
        const int baseCount = s.csrHeavyCellTaskCount[c];
        for (int local = threadIdx.x; local < baseCount; local += blockDim.x)
        {
            const int task = baseFirst + local;
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
        const int injectionFirst = s.csrHeavyInjectionCellTaskStart[c];
        const int injectionCount = s.csrHeavyInjectionCellTaskCount[c];
        for
        (
            int local = threadIdx.x;
            local < injectionCount;
            local += blockDim.x
        )
        {
            const int task = injectionFirst + local;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyInjectionPartials
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
