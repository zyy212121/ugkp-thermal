#pragma once
// Free cells use the gas reference analytic target. Only cells containing a
// constrained particle retain the contact moment closure. No host decision or
// per-cell device transfer is needed for either directory layout.
template<PressureDirectory Directory, bool Compact>
__device__ bool pressureDirectoryContainsContact(const DeviceState& s,int c)
{
    bool local=false;
    if constexpr(Directory==PressureDirectory::Split) {
        for(int i=s.preBaseCellOffset[c]+threadIdx.x;i<s.preBaseCellOffset[c+1];i+=blockDim.x)
            if(i>=0 && i<s.particleCapacity && s.pStatus[i]!=0 && s.pCellId[i]==c && s.pStuck[i]!=0)local=true;
    }
    for(int pos=s.cellParticleOffset[c]+threadIdx.x;pos<s.cellParticleOffset[c+1];pos+=blockDim.x) {
        const int i=Compact?pos:s.sortedParticleIndex[pos];
        if(i>=0 && i<s.particleCapacity
            && (Compact?s.compactPStatus[i]:s.pStatus[i])!=0
            && (Compact?s.compactPCellId[i]:s.pCellId[i])==c
            && (Compact?s.compactPStuck[i]:s.pStuck[i])!=0)local=true;
    }
    return __syncthreads_or(local)!=0;
}
template<bool FullMoments, bool Split, bool Compact=false>
__global__ void applyContactAwarePressureProjectionKernel(DeviceState* sp,PressureTime dt)
{
    const int c=blockIdx.x;if(c>=sp->nCells)return;
    constexpr auto directory=Split?PressureDirectory::Split:PressureDirectory::Full;
    if(pressureDirectoryContainsContact<directory,Compact>(*sp,c))
        runCellPressureProjection<ConstrainedPressureOperation<FullMoments>,directory,Compact>(sp,dt);
    else
        runCellPressureProjection<AnalyticPressureOperation,directory,Compact>(sp,dt);
}
__global__ void markPressureContactCellsKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int n=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        if(s.pStatus[i]!=0 && s.pStuck[i]!=0 && s.pCellId[i]>=0 && s.pCellId[i]<s.nCells)
            atomicExch(s.pressureCellHasContact+s.pCellId[i],1);
}
template<bool FullMoments>
__global__ void applyContactAwarePressureUnsortedKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int n=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        if(s.pStatus[i]==0)continue;
        const int c=s.pCellId[i];if(c<0||c>=s.nCells)continue;
        if(s.pressureCellHasContact[c]) {
            applyConstrainedPressureProjectionUnsortedParticle<FullMoments>(s,i);
        } else {
            const auto q=s.pressureProjectionCache[c];if(!q.active)continue;
            applyCollisionalPressureProjectionOneParticle(s,i,c,
                q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,q.theta1,q.resolved,q.thermalScale,q.thetaScale);
        }
    }
}
template<bool FullMoments>
__global__ void publishContactPressureMomentsKernel(DeviceState* sp)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<s.nCells && s.pressureCellHasContact[c])
        publishPressureParticleMomentsDevice<FullMoments>(s,c,
            s.pressureParticleMoments+7u*static_cast<size_t>(c),s.pressureParticleCount[c]);
}
