#pragma once
template<bool WarpAggregated>
__global__ void countSplitPreInjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int begin = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int end = clampRange
    (
        *s.particleCountDevice,
        begin,
        s.particleCapacity
    );
    for
    (
        int i = begin + blockIdx.x*blockDim.x + threadIdx.x;
        i < end;
        i += blockDim.x*gridDim.x
    )
    {
        const int c = s.pCellId[i];
        const bool valid = s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
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

template<bool WarpAggregated>
__global__ void scatterSplitPreInjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int begin = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int end = clampRange
    (
        *s.particleCountDevice,
        begin,
        s.particleCapacity
    );
    for
    (
        int i = begin + blockIdx.x*blockDim.x + threadIdx.x;
        i < end;
        i += blockDim.x*gridDim.x
    )
    {
        const int c = s.pCellId[i];
        const bool valid = s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
                if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
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
            if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
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
