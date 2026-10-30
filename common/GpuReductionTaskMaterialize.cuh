#pragma once
__global__ void materializeCsrReductionTasksKernel
(
    DeviceState* sp,
    const int GPU_DIRECTORY_SELECTOR
)
{
    DeviceState& s=*sp;
    const int total=s.csrCellTaskOffset[s.nCells];
    const int thread=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    if(thread==0)*s.csrHeavyTaskCount=total;
    // The cell list is metadata, independent of descriptor production. Count
    // and scan already completed; readiness is still published by the caller.
    for(int c=thread;c<s.nCells;c+=stride)if(s.csrCellTaskCount[c]>1){
        const int index=atomicAdd(s.csrHeavyCellCount,1);
        if(index<0||index>=s.nCells)asm("trap;");
        s.csrMultiTaskCellList[index]=c;
    }
    // Distribute descriptors rather than assigning a concentrated cell's
    // complete directory to one lane. Upper-bound lookup also handles empty
    // cells/repeated offsets without a host download of the scan total.
    for(int task=thread;task<total;task+=stride){
        int lo=0,hi=s.nCells;
        while(lo+1<hi){const int mid=lo+(hi-lo)/2;
            if(s.csrCellTaskOffset[mid]<=task)lo=mid;else hi=mid;}
        const int c=lo,local=task-s.csrCellTaskOffset[c],tile=s.csrHeavyTileParticles;
        if(local<0||local>=s.csrCellTaskCount[c]||tile<=0)asm("trap;");
        if(GPU_DIRECTORY_SELECTOR==GPU_DIRECTORY_FULL){
            const int begin=s.cellParticleOffset[c]+local*tile,end=s.cellParticleOffset[c+1];
            writeCsrReductionTask(s,task,c,begin,begin+min(tile,end-begin),CsrReductionTaskSource::fullIndexed);
            continue;
        }
        const int baseBegin=s.preBaseCellOffset[c],baseEnd=s.preBaseCellOffset[c+1];
        const int injectionBegin=s.cellParticleOffset[c],injectionEnd=s.cellParticleOffset[c+1];
#if GPU_DIRECTORY_HAS_BASE_ONLY
        if(GPU_DIRECTORY_SELECTOR==static_cast<int>(HeavyDirectoryKind::splitBaseAndInjection)){
#endif
            const int begin=local*tile,end=(baseEnd-baseBegin)+(injectionEnd-injectionBegin);
            writeCsrReductionTask(s,task,c,begin,begin+min(tile,end-begin),CsrReductionTaskSource::splitLogical);
#if GPU_DIRECTORY_HAS_BASE_ONLY
            continue;
        }
        const int baseCount=baseEnd-baseBegin,baseTasks=baseCount>0?1+(baseCount-1)/tile:0;
        if(local<baseTasks){
            const int begin=baseBegin+local*tile;
            writeCsrReductionTask(s,task,c,begin,begin+min(tile,baseEnd-begin),CsrReductionTaskSource::splitBaseDirect);
        }else{
            const int begin=injectionBegin+(local-baseTasks)*tile;
            writeCsrReductionTask(s,task,c,begin,begin+min(tile,injectionEnd-begin),CsrReductionTaskSource::splitInjectionIndexed);
        }
#endif
    }
}
