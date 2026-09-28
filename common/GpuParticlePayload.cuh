#pragma once
// Global copy schedule of the same field operation used by fused moment workers.
template<bool UseSurvivorDirectory = false>
__global__ void gatherCellLocalParticlePayloadKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int count = s.compactCellOffset[s.nCells];
    for (int dst = blockIdx.x*blockDim.x + threadIdx.x;
         dst < count; dst += gridDim.x*blockDim.x)
    {
        const int i = UseSurvivorDirectory
            ? s.sortedParticleIndex[dst] : s.compactPStatus[dst];
        if (i < 0 || i >= s.particleCapacity || s.pStatus[i] != 1)
        {
            asm("trap;");
        }
        const int c = s.pCellId[i];
        copyCellLocalParticle(s, i, c, dst);
    }
}


// Both scheduling policies reach this commit boundary. Fused mode emits no work.
template<bool Fused>
int launchDeferredSurvivorPayload(DeviceState* s, const int grid, const int threads)
{
    if constexpr (!Fused)
    {
        gatherCellLocalParticlePayloadKernel<true>
            <<<grid > 0 ? grid : 1, threads>>>(s->deviceState);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("gatherCellLocalParticlePayloadKernel launch", err);
            return 1;
        }
    }
    return 0;
}
