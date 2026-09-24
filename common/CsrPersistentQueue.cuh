#pragma once
// One block-cooperative queue protocol for gas, FSH and CHT.
// count is immutable during this launch. prepare runs on thread 0 and may
// reject an implicit empty slot. execute runs on every thread in the block.
// A rejected/zero-work task must not terminate the worker or bypass the
// final barrier: shared descriptors/partials are reused by the next task.
template<class State, class Operation>
__device__ __forceinline__ void runCsrPersistentQueue
(State& s, const int count, Operation& operation)
{
    __shared__ int task;
    __shared__ int valid;
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            valid = task < count && operation.prepare(s, task);
        }
        __syncthreads();
        if (task >= count) return;
        if (valid) operation.execute(s, task);
        __syncthreads();
    }
}

// Reset before EVERY consumer, including when its directory is reused.
// The reset and launch use the same CUDA stream (the default stream here).
template<class State>
inline cudaError_t resetCsrPersistentQueue(State* s)
{
    return cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
}
