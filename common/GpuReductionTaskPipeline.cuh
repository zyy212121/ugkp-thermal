#pragma once
// One task-production protocol. Directory and publication timing are explicit policies.
int prepareCsrSegmentedReductionTasks(DeviceState* s, const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR)
{
    // Every producer marks new directory contents, even while L1 is active.
    s->csrTasksReady = 0;
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
#if GPU_DIRECTORY_OWNS_TILE_POLICY
    s->csrReductionDirectoryKind = static_cast<int>(GPU_DIRECTORY_SELECTOR);
#endif
    cudaError_t err = cudaSuccess;
    const int countGrid = (s->nCells + 1 + block - 1)/block;
    countCsrReductionTasksKernel<<<countGrid, block>>>(s->deviceState, GPU_DIRECTORY_ARGUMENT);
    err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("countCsrReductionTasksKernel launch", err); return 1; }
    err = cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage, s->cellScanTempBytes,
        s->csrCellTaskCount, s->csrCellTaskOffset, s->nCells + 1);
    if (err != cudaSuccess) { setLastError("CSR cell task count exclusive scan", err); return 1; }
    const int cellGrid = (s->nCells + block - 1)/block;
    materializeCsrReductionTasksKernel<<<cellGrid, block>>>(s->deviceState, GPU_DIRECTORY_ARGUMENT);
    err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("materializeCsrReductionTasksKernel launch", err); return 1; }
    s->csrPreparedDirectoryKind = GPU_DIRECTORY_ARGUMENT;
    s->csrTasksReady = 1;
    return 0;
}
