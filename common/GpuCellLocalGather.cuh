#pragma once
// Included after the solver-specific copyCellLocalParticle hook.
// The ordered filtered path and the proven all-live path are shared with gas.
template<int BlockThreads, bool IndexOnly>
__device__ void gatherCellLocalRangeImpl
(
    DeviceState& s, const int c, const int start,
    const int end, const int outputStart, const bool allKept
)
{
    // The moment reduction counts exactly the gather predicate. Pressure
    // projection changes velocities only, so equality proves every entry lives.
    if (allKept)
    {
        for (int pos = start + threadIdx.x; pos < end; pos += BlockThreads)
        {
            if constexpr (IndexOnly)
                s.compactPStatus[outputStart + pos - start] = s.sortedParticleIndex[pos];
            else
                copyCellLocalParticle(s, s.sortedParticleIndex[pos], c,
                    outputStart + pos - start);
        }
        return;
    }
    using BlockScan = cub::BlockScan<int, BlockThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;

    int tileOutputOffset = 0;

    for
    (
        int tileStart = start;
        tileStart < end;
        tileStart += BlockThreads
    )
    {
        const int pos = tileStart + threadIdx.x;
        const int i = pos < end ? s.sortedParticleIndex[pos] : -1;
        const int keep =
            i >= 0
         && i < s.particleCapacity
         && s.pStatus[i] == 1
         && s.pCellId[i] == c;

        int localOffset = 0;
        int tileCount = 0;
        BlockScan(scanStorage).ExclusiveSum(keep, localOffset, tileCount);

        if (keep != 0)
        {
            const int dst = outputStart + tileOutputOffset + localOffset;
            if constexpr (IndexOnly) s.compactPStatus[dst] = i;
            else copyCellLocalParticle(s, i, c, dst);
        }

        tileOutputOffset += tileCount;
        __syncthreads();
    }
}

template<int BlockThreads>
__device__ void gatherCellLocalRange(DeviceState& s, const int c, const int start,
    const int end, const int outputStart, const bool allKept)
{
    gatherCellLocalRangeImpl<BlockThreads, false>(s, c, start, end, outputStart, allKept);
}

template<int BlockThreads>
__global__ void gatherCellLocalParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells || blockDim.x != BlockThreads) return;
    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    gatherCellLocalRange<BlockThreads>(s, c, start, end,
        s.compactCellOffset[c], s.cellParticleCount[c] == end - start);
}
