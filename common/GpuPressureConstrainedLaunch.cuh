
#pragma once
#include "GpuPressureFlatLayout.cuh"
template<bool FullMoments>
struct ConstrainedPressureLaunch
{
    static constexpr bool compactBeforeSplit=false;
    static cudaError_t limit(DeviceState* s,PressureTime dt,int grid,int block)
    {
#if defined(UGKWP_GPU_REAL_BITS) && UGKWP_GPU_REAL_BITS == 32
        const int scratch=1;
#else
        const int scratch=s->csrCellLocalPathEnabled==0;
#endif
        accumulateCollisionalPressureKickByCellKernel<FullMoments><<<grid,block>>>
            (s->deviceState,dt,1,scratch);
        return cudaGetLastError();
    }
    static cudaError_t project(DeviceState* s,PressureTime dt,int grid,int block,int threads,bool split,bool compact)
    {
#if defined(UGKWP_GPU_REAL_BITS) && UGKWP_GPU_REAL_BITS == 32
        if(split)
        {
            // Preflight already cached the actual limited-face increment.
            launchFlatPressure<FullMoments>(s,s->deviceState,dt,FlatPressureSegment::base);
            cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
            if(s->nBoundarySources>0)
            {
                launchFlatPressure<FullMoments>(s,s->deviceState,dt,FlatPressureSegment::injection);
                err=cudaGetLastError();if(err!=cudaSuccess)return err;
            }
            publishPressureParticleMomentsKernel<FullMoments><<<grid,block>>>(s->deviceState);
        }
        else if(compact)launchFlatPressure<FullMoments,true>(s,s->deviceState,dt,FlatPressureSegment::full);
        else launchFlatPressure<FullMoments>(s,s->deviceState,dt,FlatPressureSegment::full);
#else
        const size_t shared=(FullMoments?7u:4u)*static_cast<size_t>((threads+31)/32)*sizeof(PressureReal);
        // Read the limited faces directly. Base and injection share one closure,
        // removing stale cached deltas and the split scratch/publish roundtrip.
        if(split)applyUnifiedSplitPressureKernel<FullMoments>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
        else if(compact)applyCollisionalPressureProjectionKernel<FullMoments,true>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
        else applyCollisionalPressureProjectionKernel<FullMoments>
            <<<s->nCells,threads,shared>>>(s->deviceState,dt);
#endif
        return cudaGetLastError();
    }
    static cudaError_t projectUnsorted(DeviceState* s,PressureTime dt,int grid,int block)
    {
        applyCollisionalPressureProjectionCellAtomicKernel<<<grid,block>>>(s->deviceState,dt);
        cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
        applyCollisionalPressureProjectionParticlesAtomicKernel<FullMoments>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;
        publishPressureParticleMomentsKernel<FullMoments><<<grid,block>>>(s->deviceState);
        return cudaGetLastError();
    }
};
