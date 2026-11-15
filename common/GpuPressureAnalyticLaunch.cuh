
#pragma once
#include "GpuParticleExecutionContext.H"
struct AnalyticPressureLaunch
{
    static cudaError_t limit(DeviceState* s,PressureTime dt,int grid,int block)
    {
        accumulateCollisionalPressureKickByCellKernel<<<grid,block,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt,1);
        return cudaGetLastError();
    }
    static cudaError_t project(DeviceState* s,PressureTime dt,int,int,int threads,bool split,bool compact)
    {
        const int grid=(s->nCells+threads-1)/threads;
        preparePressureProjectionCacheKernel<<<grid,threads,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        auto err=cudaGetLastError();if(err!=cudaSuccess)return err;
        if(compact)applyCachedPressureProjectionParticlesKernel<true>
            <<<s->particleWorkGrid,s->particleBlockThreads,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState);
        else applyCachedPressureProjectionParticlesKernel<false>
            <<<s->particleWorkGrid,s->particleBlockThreads,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState);
        return cudaGetLastError();
    }
    static cudaError_t projectUnsorted(DeviceState* s,PressureTime dt,int grid,int block)
    {
        preparePressureProjectionCacheKernel<<<grid,block,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
        applyCachedPressureProjectionParticlesKernel<false><<<s->particleWorkGrid, s->particleBlockThreads,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState);
        return cudaGetLastError();
    }
};
