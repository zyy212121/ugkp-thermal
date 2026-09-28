#pragma once
// Recovery equation is an application physical adapter; task ownership is shared.
__global__ void solidRecoveryFromParticleMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    solidRecoveryFromParticleMomentsCell(s, c);
}

struct CsrMomentRecoveryOperation
{
    GPU_PIPELINE_REAL* warpPartials;
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int multiIndex)
    {
        const int c = s.csrMultiTaskCellList[multiIndex];
        finalizeCsrMomentCell(s,c,warpPartials);
        if (threadIdx.x == 0)
        {
            solidRecoveryFromParticleMomentsCell(s,c);
        }

    }
};

__global__ void finalizeCsrSegmentedMomentsAndRecoverKernel(DeviceState* sp)
{

    DeviceState& s = *sp;
    const int heavyGrid = s.multiprocessorCount < s.nCells
        ? s.multiprocessorCount : s.nCells;
    if (static_cast<int>(blockIdx.x) >= heavyGrid)
    {
        const int c = (static_cast<int>(blockIdx.x) - heavyGrid)*blockDim.x + threadIdx.x;
        if (c < s.nCells && s.csrCellTaskCount[c] <= 1)
        {
            solidRecoveryFromParticleMomentsCell(s, c);
        }
        return;
    }
    extern __shared__ GPU_PIPELINE_REAL warpPartials[];
    CsrMomentRecoveryOperation operation{warpPartials};
    runCsrPersistentQueue(s, *s.csrHeavyCellCount, operation);
}
#undef GPU_PIPELINE_REAL
