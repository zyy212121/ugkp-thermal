#pragma once
// The caller supplies scalar and physical-model constants, never runtime policy.
// Preserve native S1 initialization and L2 outputs while sharing one loop body.
template<bool HeavyReductionEnabled, bool GatherSurvivors = false>
__global__ void accumulateParticleMomentsSegmentedKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells) return;
    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
#if GPU_MOMENT_THERMAL
    // Retain the existing thermal entry guard and native instruction layout.
    if (s.csrHeavyReductionEnabled != 0) return;
#else
    if constexpr (HeavyReductionEnabled) return;
#endif
#define GPU_MOMENT_BEGIN start
#define GPU_MOMENT_COUNT_TYPE int
#define GPU_MOMENT_COUNT survivorCount
#include "GpuParticleMomentRange.inl"
#undef GPU_MOMENT_BEGIN
#undef GPU_MOMENT_COUNT_TYPE
#undef GPU_MOMENT_COUNT
    GPU_MOMENT_REAL sums[8] =
    {
        rho, momX, momY, momZ, energy, diameter, heat,
        static_cast<GPU_MOMENT_REAL>(survivorCount)
    };
    extern __shared__ GPU_MOMENT_REAL warpPartials[];
    blockReduceComponentSums<8>(sums, warpPartials);
    if (threadIdx.x == 0)
    {
        s.cellParticleCount[c] = static_cast<int>(sums[7]);
        if (c == 0)
        {
            s.cellParticleCount[s.nCells] = 0;
        }
        const GPU_MOMENT_REAL invV = GPU_MOMENT_R(1.0)/clampMin(s.V[c], s.rhoMin);
        s.momRhoP[c] = sums[0]*invV;
        s.momRhoUPx[c] = sums[1]*invV;
        s.momRhoUPy[c] = sums[2]*invV;
        s.momRhoUPz[c] = sums[3]*invV;
        s.momRhoEP[c] = sums[4]*invV;
        s.momRhoPD[c] = sums[5]*invV;
        s.momRhoHpP[c] = sums[6]*invV;
    }
}

template<bool GatherSurvivors = false>
__device__ void accumulateCsrHeavyMomentTask
(
    DeviceState& s, const int c, const int begin, const int end,
    GPU_MOMENT_REAL (&sums)[8], GPU_MOMENT_REAL* warpPartials
)
{
#define GPU_MOMENT_BEGIN begin
#if GPU_MOMENT_THERMAL
#define GPU_MOMENT_COUNT_TYPE GPU_MOMENT_REAL
#else
#define GPU_MOMENT_COUNT_TYPE int
#endif
#define GPU_MOMENT_COUNT count
#include "GpuParticleMomentRange.inl"
#undef GPU_MOMENT_BEGIN
#undef GPU_MOMENT_COUNT_TYPE
#undef GPU_MOMENT_COUNT
    sums[0] = rho;
    sums[1] = momX;
    sums[2] = momY;
    sums[3] = momZ;
    sums[4] = energy;
    sums[5] = diameter;
    sums[6] = heat;
    sums[7] = static_cast<GPU_MOMENT_REAL>(count);
    blockReduceComponentSums<8>(sums, warpPartials);
}
#undef GPU_MOMENT_REAL
#undef GPU_MOMENT_R
#undef GPU_MOMENT_THERMAL
#undef GPU_MOMENT_GATHER
