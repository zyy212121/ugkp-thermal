#pragma once
// Shared physical update; the including backend supplies its state and scalar adapters.
template<class State> __device__ GpuReal gasWallExposedAreaFractionImpl(const State&,int,long){return GPU_R(1.0);}

template<class State>
__device__ auto gasWallExposedAreaFractionImpl
(
    const State& s,
    const int f, int
) -> decltype(s.particleWallRepresentedContactArea, s.particleWallContactAreaScale, GPU_R(1.0))
{
    if
    (
        f < s.nInternalFaces
     || f >= s.nFaces
     || s.particleStuckModelConfigured == 0
     || !s.particlesMayBePresent
     || s.particleStuckCandidateMask == nullptr
     || s.particleStuckCandidateMask[f] == 0
    )
    {
        return GPU_R(1.0);
    }
    if
    (
        s.particleWallRepresentedContactArea == nullptr
     || s.particleWallContactAreaScale == nullptr
    )
    {
        asm("trap;");
        return GPU_R(0.0);
    }

    GpuReal exposed=GPU_R(1.0);
    if(!ugkwp::wallHeatExposedFraction(s.magSf[f],s.particleWallRepresentedContactArea[f],
        s.particleWallContactAreaScale[f],exposed))
    {
        asm("trap;");
        return GPU_R(0.0);
    }
    return exposed;
}
template<class State>
__device__ GpuReal gasWallExposedAreaFraction(const State& s,int face)
{return gasWallExposedAreaFractionImpl(s,face,0);}
