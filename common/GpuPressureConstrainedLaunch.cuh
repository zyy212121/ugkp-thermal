
#pragma once
#include "GpuParticleExecutionContext.H"
template<bool FullMoments>
struct ConstrainedPressureLaunch
{
    static cudaError_t limit(DeviceState* s,PressureTime dt,int grid,int block)
    {
        const int scratch=s->csrCellLocalPathEnabled==0;
        accumulateCollisionalPressureKickByCellKernel<FullMoments><<<grid,block,0,ugkp::gpu::particleRecordingStream(s)>>>
            (s->deviceState,dt,1,scratch);
        return cudaGetLastError();
    }
    static cudaError_t project(DeviceState* s,PressureTime dt,int grid,int block,int threads,bool split,bool compact)
    {
        const size_t shared=(FullMoments?7u:4u)*static_cast<size_t>((threads+31)/32)*sizeof(PressureReal);
        // Read the limited faces directly. Base and injection share one closure,
        // removing stale cached deltas and the split scratch/publish roundtrip.
        if(split)applyUnifiedSplitPressureKernel<FullMoments>
            <<<s->nCells,threads,shared,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        else if(compact)applyCollisionalPressureProjectionKernel<FullMoments,true>
            <<<s->nCells,threads,shared,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        else applyCollisionalPressureProjectionKernel<FullMoments>
            <<<s->nCells,threads,shared,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        return cudaGetLastError();
    }
    static cudaError_t projectUnsorted(DeviceState* s,PressureTime dt,int grid,int block)
    {
        applyCollisionalPressureProjectionCellAtomicKernel<<<grid,block,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState,dt);
        cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
        applyCollisionalPressureProjectionParticlesAtomicKernel<FullMoments>
            <<<s->particleWorkGrid, s->particleBlockThreads,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;
        publishPressureParticleMomentsKernel<FullMoments><<<grid,block,0,ugkp::gpu::particleRecordingStream(s)>>>(s->deviceState);
        return cudaGetLastError();
    }
};
