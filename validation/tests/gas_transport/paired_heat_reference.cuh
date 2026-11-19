#pragma once


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
