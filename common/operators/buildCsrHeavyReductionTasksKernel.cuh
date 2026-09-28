#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void buildCsrHeavyReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int begin = s.cellParticleOffset[c];
        const int end = s.cellParticleOffset[c + 1];
        const int count = end - begin;
        if (count <= s.dynamicHeavyThreshold)
        {
            continue;
        }

        const int nTasks =
            1 + (count - 1)/s.csrHeavyTileParticles;
        const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
        if
        (
            taskStart < 0
         || nTasks <= 0
         || taskStart > s.csrHeavyTaskCapacity - nTasks
        )
        {
            asm("trap;");
        }

        s.csrHeavyCellTaskStart[c] = taskStart;
        s.csrHeavyCellTaskCount[c] = nTasks;
        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;
        for (int localTask = 0; localTask < nTasks; ++localTask)
        {
            const int task = taskStart + localTask;
            const int taskBegin = begin + localTask*s.csrHeavyTileParticles;
            const int remaining = end - taskBegin;
            const int taskLength =
                remaining < s.csrHeavyTileParticles
              ? remaining
              : s.csrHeavyTileParticles;
            const int taskEnd = taskBegin + taskLength;
            s.csrHeavyTaskCell[task] = c;
            s.csrHeavyTaskBegin[task] = taskBegin;
            s.csrHeavyTaskEnd[task] = taskEnd;
        }
    }
}

__global__ void buildSplitCsrHeavyReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.csrHeavyCellTaskStart[c] = 0;
        s.csrHeavyCellTaskCount[c] = 0;
        s.csrHeavyInjectionCellTaskStart[c] = 0;
        s.csrHeavyInjectionCellTaskCount[c] = 0;

        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        const int baseCount = baseEnd - baseBegin;
        const int injectionCount = injectionEnd - injectionBegin;
        if (baseCount + injectionCount <= s.dynamicHeavyThreshold)
        {
            continue;
        }

        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;

        if (baseCount > 0)
        {
            const int nTasks = 1 + (baseCount - 1)/s.csrHeavyTileParticles;
            const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
            if
            (
                taskStart < 0 || nTasks <= 0
             || taskStart > s.csrHeavyTaskCapacity - nTasks
            )
            {
                asm("trap;");
            }
            s.csrHeavyCellTaskStart[c] = taskStart;
            s.csrHeavyCellTaskCount[c] = nTasks;
            for (int localTask = 0; localTask < nTasks; ++localTask)
            {
                const int task = taskStart + localTask;
                const int begin = baseBegin + localTask*s.csrHeavyTileParticles;
                const int end = min(begin + s.csrHeavyTileParticles, baseEnd);
                s.csrHeavyTaskCell[task] = c;
                s.csrHeavyTaskBegin[task] = begin;
                s.csrHeavyTaskEnd[task] = end;
            }
        }

        if (injectionCount > 0)
        {
            const int nTasks =
                1 + (injectionCount - 1)/s.csrHeavyTileParticles;
            const int taskStart =
                atomicAdd(s.csrHeavyInjectionTaskCount, nTasks);
            if
            (
                taskStart < 0 || nTasks <= 0
             || taskStart > s.csrHeavyTaskCapacity - nTasks
            )
            {
                asm("trap;");
            }
            s.csrHeavyInjectionCellTaskStart[c] = taskStart;
            s.csrHeavyInjectionCellTaskCount[c] = nTasks;
            for (int localTask = 0; localTask < nTasks; ++localTask)
            {
                const int task = taskStart + localTask;
                const int begin =
                    injectionBegin + localTask*s.csrHeavyTileParticles;
                const int end =
                    min(begin + s.csrHeavyTileParticles, injectionEnd);
                s.csrHeavyInjectionTaskCell[task] = c;
                s.csrHeavyInjectionTaskBegin[task] = begin;
                s.csrHeavyInjectionTaskEnd[task] = end;
            }
        }
    }
}
