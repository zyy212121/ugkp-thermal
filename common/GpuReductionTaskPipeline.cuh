#pragma once
// Task mathematics and scan order are identical in direct and conditional execution.
int submitCsrSegmentedReductionTasks(DeviceState* s,const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR,cudaStream_t stream)
{
    const int countGrid=(s->nCells+1+block-1)/block;
    countCsrReductionTasksKernel<<<countGrid,block,0,stream>>>(s->deviceState,GPU_DIRECTORY_ARGUMENT);
    auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("countCsrReductionTasksKernel launch",err);return 1;}
    err=cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage,s->cellScanTempBytes,
        s->csrCellTaskCount,s->csrCellTaskOffset,s->nCells+1,stream);
    if(err!=cudaSuccess){setLastError("CSR cell task count exclusive scan",err);return 1;}
    const int cellGrid=(s->nCells+block-1)/block;
    materializeCsrReductionTasksKernel<<<cellGrid,block,0,stream>>>(s->deviceState,GPU_DIRECTORY_ARGUMENT);
    err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("materializeCsrReductionTasksKernel launch",err);return 1;}
    return 0;
}
int prepareCsrSegmentedReductionTasksImpl(DeviceState* s,const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR,bool inspectionOnly)
{
    s->csrTasksReady=0;
    if(s->csrHeavyReductionEnabled==0||s->particleCapacity<=0)return 0;
#if GPU_DIRECTORY_OWNS_TILE_POLICY
    s->csrReductionDirectoryKind=static_cast<int>(GPU_DIRECTORY_SELECTOR);
#endif
#ifdef UGKP_RESIDENT_CSR_CHOICE
    const int result=launchCsrConditional(s,GPU_DIRECTORY_ARGUMENT,block,GPU_DIRECTORY_ARGUMENT,0,
        [&](cudaStream_t stream,bool heavy){
            if(heavy)return submitCsrSegmentedReductionTasks(s,block,GPU_DIRECTORY_SELECTOR,stream);
            auto err=cudaMemsetAsync(s->csrHeavyTaskCount,0,sizeof(int),stream);
            if(err==cudaSuccess)err=cudaMemsetAsync(s->csrHeavyCellCount,0,sizeof(int),stream);
            if(err!=cudaSuccess){setLastError("clear inactive automatic CSR schedule",err);return 1;}
            return 0;
        },inspectionOnly);
#else
    const int result=submitCsrSegmentedReductionTasks(s,block,GPU_DIRECTORY_SELECTOR,ugkp::gpu::particleRecordingStream(s));
#endif
    if(result)return result;
    s->csrPreparedDirectoryKind=GPU_DIRECTORY_ARGUMENT;s->csrTasksReady=1;
    return 0;
}

int prepareCsrSegmentedReductionTasks(DeviceState* s,const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR){
    return prepareCsrSegmentedReductionTasksImpl(s,block,GPU_DIRECTORY_SELECTOR,false);
}
int prepareCsrSegmentedReductionTasksOnInspection(DeviceState* s,const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR){
    return prepareCsrSegmentedReductionTasksImpl(s,block,GPU_DIRECTORY_SELECTOR,true);
}
