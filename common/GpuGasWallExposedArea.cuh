#pragma once
// Shared physical update; the including backend supplies its state and scalar adapters.
template<class State> __device__ GpuReal gasWallExposedAreaFraction(const State&,int){return GPU_R(1.0);}

__device__ GpuReal gasWallExposedAreaFraction
(
    const DeviceState& s,
    const int f
)
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

    const GpuReal faceArea = s.magSf[f];
    const GpuReal representedArea =
        s.particleWallRepresentedContactArea[f];
    const GpuReal contactAreaScale =
        s.particleWallContactAreaScale[f];
    if
    (
        !finiteDevice(faceArea)
     || !finiteDevice(representedArea)
     || !finiteDevice(contactAreaScale)
     || !(faceArea > GPU_R(0.0))
     || representedArea < GPU_R(0.0)
     || !(contactAreaScale > GPU_R(0.0))
     || contactAreaScale > GPU_R(1.0)
    )
    {
        asm("trap;");
        return GPU_R(0.0);
    }

    const GpuReal occupiedArea = representedArea*contactAreaScale;
    if (!finiteDevice(occupiedArea) || occupiedArea < GPU_R(0.0))
    {
        asm("trap;");
        return GPU_R(0.0);
    }
    return clampRange(GPU_R(1.0) - occupiedArea/faceArea, GPU_R(0.0), GPU_R(1.0));
}
