#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void maximumDirectoryOccupancyKernel
(
    DeviceState* sp,
    const int splitPreDirectoryActive,
    int* maximumOccupancy
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        if (splitPreDirectoryActive != 0)
        {
            count += s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
        }
        atomicMax(maximumOccupancy, count);
    }
}
