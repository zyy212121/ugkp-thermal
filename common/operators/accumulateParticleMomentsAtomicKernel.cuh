#pragma once
#include "GpuMaterialEnthalpyMoment.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void accumulateParticleMomentsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
#if !GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL heatFactor = particleHeatFactorDevice(s);
#endif
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
#define GPU_MOMENT_REAL GPU_OPERATOR_REAL
#define GPU_MOMENT_R(x) GPU_OPERATOR_R(x)
#define GPU_MOMENT_THERMAL GPU_OPERATOR_THERMAL
#include "GpuParticleMomentContribution.inl"
#undef GPU_MOMENT_THERMAL
#undef GPU_MOMENT_R
#undef GPU_MOMENT_REAL
        atomicAdd(&s.momRhoP[c], m);
        atomicAdd(&s.momRhoUPx[c], m*ux);
        atomicAdd(&s.momRhoUPy[c], m*uy);
        atomicAdd(&s.momRhoUPz[c], m*uz);
        atomicAdd
        (
            &s.momRhoEP[c],
            particleEnergy
        );
        atomicAdd(&s.momRhoPD[c], m*d);
#if GPU_OPERATOR_THERMAL
        atomicAdd(&s.momRhoHpP[c], particleHeat);
#else
        if (heatFactor > GPU_OPERATOR_R(0.0)) atomicAdd(&s.momRhoHpP[c], particleHeat);
#endif
        atomicAdd(&s.cellParticleCount[c], 1);
    }
}
