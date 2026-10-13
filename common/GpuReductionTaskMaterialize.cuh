#pragma once
__global__ void materializeCsrReductionTasksKernel
(
    DeviceState* sp,
    const int GPU_DIRECTORY_SELECTOR
)
{
    DeviceState& s = *sp;
    // The scan is complete. Publish its total within the existing producer;
    // consumers launch after this kernel, so no separate publication is needed.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *s.csrHeavyTaskCount = s.csrCellTaskOffset[s.nCells];
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int taskStart = s.csrCellTaskOffset[c];
        const int nTasks = s.csrCellTaskCount[c];
        if (nTasks == 0)
        {
            continue;
        }
        if (taskStart < 0 || taskStart > s.csrHeavyTaskCapacity - nTasks)
        {
            asm("trap;");
        }
        if (nTasks > 1)
        {
            const int multiIndex = atomicAdd(s.csrHeavyCellCount, 1);
            if (multiIndex < 0 || multiIndex >= s.nCells)
            {
                asm("trap;");
            }
            s.csrMultiTaskCellList[multiIndex] = c;
        }

        const int tile = s.csrHeavyTileParticles;
        int localTask = 0;
        if (GPU_DIRECTORY_SELECTOR == GPU_DIRECTORY_FULL)
        {
            const int begin = s.cellParticleOffset[c];
            const int end = s.cellParticleOffset[c + 1];
            for (; localTask < nTasks; ++localTask)
            {
                const int taskBegin = begin + localTask*tile;
                writeCsrReductionTask
                (
                    s, taskStart + localTask, c, taskBegin,
                    taskBegin + min(tile, end - taskBegin),
                    CsrReductionTaskSource::fullIndexed
                );
            }
            continue;
        }
        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        const int baseCount = baseEnd - baseBegin;
        const int injectionCount = injectionEnd - injectionBegin;
#if GPU_DIRECTORY_HAS_BASE_ONLY
        if
        (
            GPU_DIRECTORY_SELECTOR == static_cast<int>
            (
                HeavyDirectoryKind::splitBaseAndInjection
            )
        )
        {
#endif
            const int totalCount = baseCount + injectionCount;
            for
            (
                int taskBegin = 0;
                taskBegin < totalCount;
                taskBegin += min(tile, totalCount - taskBegin)
            )
            {
                writeCsrReductionTask
                (
                    s, taskStart + localTask++, c, taskBegin,
                    taskBegin + min(tile, totalCount - taskBegin),
                    CsrReductionTaskSource::splitLogical
                );
            }
            if (localTask != nTasks)
            {
                asm("trap;");
            }
#if GPU_DIRECTORY_HAS_BASE_ONLY
            continue;
        }
        for (int taskBegin = baseBegin; taskBegin < baseEnd; taskBegin += min(tile, baseEnd - taskBegin))
        {
            writeCsrReductionTask
            (
                s, taskStart + localTask++, c, taskBegin,
                taskBegin + min(tile, baseEnd - taskBegin),
                CsrReductionTaskSource::splitBaseDirect
            );
        }
        if (GPU_DIRECTORY_SELECTOR == static_cast<int>(HeavyDirectoryKind::baseOnly))
        {
            continue;
        }
        for
        (
            int taskBegin = injectionBegin;
            taskBegin < injectionEnd;
            taskBegin += min(tile, injectionEnd - taskBegin)
        )
        {
            writeCsrReductionTask
            (
                s, taskStart + localTask++, c, taskBegin,
                taskBegin + min(tile, injectionEnd - taskBegin),
                CsrReductionTaskSource::splitInjectionIndexed
            );
        }
        if (localTask != nTasks)
        {
            asm("trap;");
        }
#endif
    }
}
