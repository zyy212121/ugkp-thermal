#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool WarpAggregated, bool SurvivorsOnly = false>
__global__ void countParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        const int c = s.pCellId[i];
        const bool valid = (SurvivorsOnly ? s.pStatus[i] == 1 : s.pStatus[i] != 0) && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                atomicAdd(&s.cellParticleCount[c], 1);
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            if (lane == leader)
            {
                atomicAdd(&s.cellParticleCount[c], __popc(groupMask));
            }
        }
#else
        if (valid)
        {
            atomicAdd(&s.cellParticleCount[c], 1);
        }
#endif
    }
}

__global__ void initialiseParticleCellWritesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.cellParticleWrite[c] = s.cellParticleOffset[c];
    }
}

template<bool WarpAggregated, bool SurvivorsOnly = false>
__global__ void scatterParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        const int c = s.pCellId[i];
        const bool valid = (SurvivorsOnly ? s.pStatus[i] == 1 : s.pStatus[i] != 0) && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
                if
                (
                    pos >= s.cellParticleOffset[c]
                 && pos < s.cellParticleOffset[c + 1]
                )
                {
                    s.sortedParticleIndex[pos] = i;
                }
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            int groupBase = 0;
            if (lane == leader)
            {
                groupBase = atomicAdd(&s.cellParticleWrite[c], __popc(groupMask));
            }
            groupBase = __shfl_sync(groupMask, groupBase, leader);
            const unsigned int lowerLaneMask = lane == 0 ? 0u : ((1u << lane) - 1u);
            const int laneRank = __popc(groupMask & lowerLaneMask);
            const int pos = groupBase + laneRank;
            if
            (
                pos >= s.cellParticleOffset[c]
             && pos < s.cellParticleOffset[c + 1]
            )
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#else
        if (valid)
        {
            const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
            if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#endif
    }
}
