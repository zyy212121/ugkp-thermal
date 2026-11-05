#pragma once
// Gas reference projection: one cell target followed by parcel velocity mapping.
#include "GpuPressureUnsortedAlgebra.cuh"
#include "GpuPressureParticleUpdate.cuh"
template<bool CompactParticles = false>
__device__ inline void applyCollisionalPressureProjectionOneParticle
(
    DeviceState& s,
    const int i,
    const int c,
    const double ux0,
    const double uy0,
    const double uz0,
    const double ux1,
    const double uy1,
    const double uz1,
    const double theta1,
    const bool resolved,
    const double thermalScale,
    const double thetaScale
)
{
    if
    (
        i < 0
     || i >= s.particleCapacity
     || (CompactParticles ? s.compactPStatus : s.pStatus)[i] == 0
     || (CompactParticles ? s.compactPCellId : s.pCellId)[i] != c
    )
    {
        return;
    }
    updateMobilePressureParticle<CompactParticles>
        (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
}

#include "GpuPreparePressureProjection.cuh"

#include "GpuPressureCellTraversal.cuh"
#include "GpuPressureAnalyticAdapter.cuh"
template<bool SplitDirectory, bool CompactParticles = false>
__global__ void applyAnalyticPressureProjectionKernel(DeviceState* sp,const double dt)
{
    runCellPressureProjection<AnalyticPressureOperation,
        SplitDirectory?PressureDirectory::Split:PressureDirectory::Full,CompactParticles>(sp,dt);
}

__global__ void preparePressureProjectionCacheKernel
(DeviceState* sp, const double kickDt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < s.nCells)
        s.pressureProjectionCache[c] = preparePressureProjectionCell(s, c, kickDt);
}

__global__ void applyCachedPressureProjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        if (s.pStatus[i] == 0) continue;
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells) continue;
        const PressureProjectionCell q = s.pressureProjectionCache[c];
        if (!q.active) continue;
        applyCollisionalPressureProjectionOneParticle(s, i, c,
            q.ux0, q.uy0, q.uz0, q.ux1, q.uy1, q.uz1, q.theta1,
            q.resolved, q.thermalScale, q.thetaScale);
    }
}
