#pragma once





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



template<class State>
inline cudaError_t resetCsrPersistentQueue(State* s,cudaStream_t stream=nullptr)
{
    return cudaMemsetAsync(s->csrHeavyTaskCursor, 0, sizeof(int),stream);
}
