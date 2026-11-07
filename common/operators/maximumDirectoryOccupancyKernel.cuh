#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void maximumDirectoryOccupancyKernel
(
    DeviceState* sp,
    const int GPU_DIRECTORY_SELECTOR,
    int* maximumOccupancy,
    const bool scheduledInspection=false
)
{
    DeviceState& s = *sp;
#ifdef UGKP_RESIDENT_CSR_CHOICE
    if(scheduledInspection&&!s.csrAutomaticRuntime->inspect)return;
#endif
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
#if GPU_DIRECTORY_HAS_BASE_ONLY
        if (GPU_DIRECTORY_SELECTOR == static_cast<int>(HeavyDirectoryKind::baseOnly))
            count = s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
        else
#endif
        if (GPU_DIRECTORY_SELECTOR != GPU_DIRECTORY_FULL)
            count += s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
        atomicMax(maximumOccupancy, count);
    }
}
