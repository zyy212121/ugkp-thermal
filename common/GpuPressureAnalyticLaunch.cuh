
#pragma once
struct AnalyticPressureLaunch
{
    static cudaError_t limit(DeviceState* s,PressureTime dt,int grid,int block)
    {
        accumulateCollisionalPressureKickByCellKernel<<<grid,block>>>(s->deviceState,dt,1);
        return cudaGetLastError();
    }
    static cudaError_t project(DeviceState* s,PressureTime dt,int,int,int threads,bool split,bool compact)
    {
        if(compact)applyCollisionalPressureProjectionKernel<false,true><<<s->nCells,threads>>>(s->deviceState,dt);
        else if(split)applyCollisionalPressureProjectionKernel<true><<<s->nCells,threads>>>(s->deviceState,dt);
        else applyCollisionalPressureProjectionKernel<false><<<s->nCells,threads>>>(s->deviceState,dt);
        return cudaGetLastError();
    }
    static cudaError_t projectUnsorted(DeviceState* s,PressureTime dt,int grid,int block)
    {
        preparePressureProjectionCacheKernel<<<grid,block>>>(s->deviceState,dt);
        cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
        applyCachedPressureProjectionParticlesKernel<<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        return cudaGetLastError();
    }
};
