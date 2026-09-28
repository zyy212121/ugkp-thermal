#pragma once
// Directory representation and producer fusion are compile-time scheduling policies.
__global__ void countCsrReductionTasksKernel(DeviceState* sp, const int GPU_DIRECTORY_SELECTOR)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c > s.nCells) return;
#if GPU_DIRECTORY_FUSED_PRODUCER
    const int tile = csrReductionTileParticles(s, GPU_DIRECTORY_SELECTOR);
    if (c == 0)
    {
        s.csrHeavyCellThreshold = tile;
        s.csrHeavyTileParticles = tile;
        *s.csrHeavyCellCount = 0;
        *s.csrHeavyTaskCount = 0;
        s.csrReductionDirectoryKind = GPU_DIRECTORY_SELECTOR;
    }
#endif
    if (c == s.nCells)
    {
        s.csrCellTaskCount[c] = 0;
        return;
    }
#if !GPU_DIRECTORY_FUSED_PRODUCER
    const int tile = s.csrHeavyTileParticles;
#endif
    if (tile <= 0) asm("trap;");
    if (GPU_DIRECTORY_SELECTOR == GPU_DIRECTORY_FULL)
    {
        const int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        s.csrCellTaskCount[c] = count > 0 ? 1 + (count - 1)/tile : 0;
        return;
    }
    const int baseCount = s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
#if GPU_DIRECTORY_HAS_BASE_ONLY
    const int baseTasks = baseCount > 0 ? 1 + (baseCount - 1)/tile : 0;
    if (GPU_DIRECTORY_SELECTOR == static_cast<int>(HeavyDirectoryKind::baseOnly))
    {
        s.csrCellTaskCount[c] = baseTasks;
        return;
    }
#endif
    const int injectionCount = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
    const int totalCount = baseCount + injectionCount;
    s.csrCellTaskCount[c] = totalCount > 0 ? 1 + (totalCount - 1)/tile : 0;
}
