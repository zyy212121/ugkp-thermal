#pragma once
template<bool FullMoments>
struct ContactAwarePressureLaunch
{
    static cudaError_t limit(DeviceState* s,PressureTime dt,int grid,int block)
    {
        auto err=ConstrainedPressureLaunch<FullMoments>::limit(s,dt,grid,block);
        if(err!=cudaSuccess || s->csrCellLocalPathEnabled)return err;
        err=cudaMemset(s->pressureCellHasContact,0,s->nCells*sizeof(int));
        if(err!=cudaSuccess)return err;
        markPressureContactCellsKernel<<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState);
        return cudaGetLastError();
    }
    static cudaError_t project(DeviceState* s,PressureTime dt,int,int,int threads,bool split,bool compact)
    {
        const size_t shared=(FullMoments?7u:4u)*static_cast<size_t>((threads+31)/32)*sizeof(PressureReal);
        if(split)applyContactAwarePressureProjectionKernel<FullMoments,true>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
        else if(compact)applyContactAwarePressureProjectionKernel<FullMoments,false,true>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
        else applyContactAwarePressureProjectionKernel<FullMoments,false>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
        return cudaGetLastError();
    }
    static cudaError_t projectUnsorted(DeviceState* s,PressureTime dt,int grid,int block)
    {
        preparePressureProjectionCacheKernel<<<grid,block>>>(s->deviceState,dt);
        auto err=cudaGetLastError();if(err!=cudaSuccess)return err;
        applyContactAwarePressureUnsortedKernel<FullMoments>
            <<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;
        publishContactPressureMomentsKernel<FullMoments><<<grid,block>>>(s->deviceState);
        return cudaGetLastError();
    }
};
