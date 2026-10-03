#pragma once
__global__ void gatherSelectedParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int selectedCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    for
    (
        int dst = blockIdx.x*blockDim.x + threadIdx.x;
        dst < selectedCount;
        dst += blockDim.x*gridDim.x
    )
    {
        const int i = s.sortedParticleIndex[dst];
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
        )
        {
            asm("trap;");
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            asm("trap;");
        }

        copyCellLocalParticle(s, i, c, dst);
    }
}
