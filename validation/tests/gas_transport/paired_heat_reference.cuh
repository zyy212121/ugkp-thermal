#pragma once
// HOST fixture launch adapters for the pre-fusion schedule. All numerical work
// remains in the same per-cell/per-particle production helpers as the fused path.
void clearPairedParticleHeatKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<s.nCells)clearPairedParticleHeatCell(s,c);
}
void reducePairedParticleHeatKernel(DeviceState* sp,GpuTime dt)
{
    DeviceState& s=*sp;
    const int n=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        accumulatePairedParticleHeatCandidate(s,i,dt);
}
