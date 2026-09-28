#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool PoissonMode>
__global__ void accumulateCsrHeavyPoolTasksPersistentKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ GPU_OPERATOR_REAL collisionProbability;
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
        if (threadIdx.x == 0)
        {
            if (PoissonMode)
            {
                const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);
                collisionProbability =
                    (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
                  ? GPU_OPERATOR_R(0.0)
                  : clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
            }
            else
            {
                collisionProbability = GPU_OPERATOR_R(1.0);
            }
        }
        __syncthreads();

        GPU_OPERATOR_REAL sums[8];
        accumulateCsrHeavyPoolTask<PoissonMode>
        (
            s,
            c,
            s.csrHeavyTaskBegin[task],
            s.csrHeavyTaskEnd[task],
            false,
            collisionProbability,
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
