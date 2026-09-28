#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool FullMoments, bool CompactParticles = false>
__global__ void applyCollisionalPressureProjectionKernel(DeviceState* sp,const GPU_OPERATOR_TIME dt)
{
    runCellPressureProjection<ConstrainedPressureOperation<FullMoments>,PressureDirectory::Full,CompactParticles>(sp,dt);
}
template<bool FullMoments>
__global__ void applyUnifiedSplitPressureKernel(DeviceState* sp,const GPU_OPERATOR_TIME dt)
{
    runCellPressureProjection<ConstrainedPressureOperation<FullMoments>,PressureDirectory::Split,false>(sp,dt);
}


template<bool DirectBase, bool FullMoments>
__global__ void applyCollisionalPressureProjectionSplitSegmentKernel(DeviceState* sp)
{
    runCellPressureProjection<ConstrainedPressureOperation<FullMoments>,
        DirectBase?PressureDirectory::Base:PressureDirectory::Injection,false>(sp,0);
}
