#pragma once
// Shared full thermal payload. Compile-time layout adapters preserve the
// existing cold1D copy form; there is no runtime policy or extra traversal.
template<bool HasContactAge, bool PackedCold1D>
struct CellLocalThermalExtraFields
{
    template<class State>
    static __device__ __forceinline__ void copyContactAge(State& s, const int i, const int dst)
    {
        if constexpr (HasContactAge) s.compactPContactAge[dst] = s.pContactAge[i];
    }
    template<class State>
    static __device__ __forceinline__ void copy(State& s, const int i, const int dst)
    {
    s.compactPStuck[dst] = s.pStuck[i];
    s.compactPStuckFaceId[dst] = s.pStuckFaceId[i];
    s.compactPDepositionArea[dst] = s.pDepositionArea[i];
    s.compactPContactDuration[dst] = s.pContactDuration[i];
    s.compactPContactMaximumArea[dst] = s.pContactMaximumArea[i];
    s.compactPContactPeakFraction[dst] = s.pContactPeakFraction[i];
    if (s.coldWallSolidificationEnabled != 0)
    {
        if constexpr (PackedCold1D)
        {
        static_assert(Foam::gpuThermal::coldWallAxialNodeCount % 4 == 0,
            "cold-wall node stride must preserve float4 alignment");
        static_assert(Foam::gpuThermal::coldWallRadialRingCount % 4 == 0,
            "cold-wall ring stride must preserve float4 alignment");


        const int nodeVectors = Foam::gpuThermal::coldWallAxialNodeCount/4;
        const int ringVectors = Foam::gpuThermal::coldWallRadialRingCount/4;
        #pragma unroll
        for (int node = 0; node < nodeVectors; ++node)
        {
            reinterpret_cast<float4*>(s.compactPColdNodeSpecificEnthalpy)
                [dst*nodeVectors + node] =
            reinterpret_cast<const float4*>(s.pColdNodeSpecificEnthalpy)
                [i*nodeVectors + node];
        }
        #pragma unroll
        for (int ring = 0; ring < ringVectors; ++ring)
        {
            reinterpret_cast<float4*>(s.compactPColdRingSolidMass)
                [dst*ringVectors + ring] =
            reinterpret_cast<const float4*>(s.pColdRingSolidMass)
                [i*ringVectors + ring];
        }
        }
        else
        {
        for
        (
            int node = 0;
            node < Foam::gpuThermal::coldWallAxialNodeCount;
            ++node
        )
        {
            s.compactPColdNodeSpecificEnthalpy
            [
                dst*Foam::gpuThermal::coldWallAxialNodeCount + node
            ] = s.pColdNodeSpecificEnthalpy
            [
                i*Foam::gpuThermal::coldWallAxialNodeCount + node
            ];
        }
        for
        (
            int ring = 0;
            ring < Foam::gpuThermal::coldWallRadialRingCount;
            ++ring
        )
        {
            s.compactPColdRingSolidMass
            [
                dst*Foam::gpuThermal::coldWallRadialRingCount + ring
            ] = s.pColdRingSolidMass
            [
                i*Foam::gpuThermal::coldWallRadialRingCount + ring
            ];
        }
        }
        s.compactPColdFrozenArea[dst] = s.pColdFrozenArea[i];
        s.compactPColdContactAge[dst] = s.pColdContactAge[i];
    }
    if (s.coldWall2DEnabled != 0)
    {
        for
        (
            int node = 0;
            node < Foam::gpuThermal::coldWall2DNodeCount;
            ++node
        )
        {
            s.compactPCold2DNodeSpecificEnthalpy
            [dst*Foam::gpuThermal::coldWall2DNodeCount + node] =
                s.pCold2DNodeSpecificEnthalpy
                [i*Foam::gpuThermal::coldWall2DNodeCount + node];
        }
        for
        (
            int ring = 0;
            ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
            ++ring
        )
        {
            s.compactPCold2DRingContactAge
            [dst*Foam::gpuThermal::coldWall2DRadialNodeCount + ring] =
                s.pCold2DRingContactAge
                [i*Foam::gpuThermal::coldWall2DRadialNodeCount + ring];
        }
        s.compactPCold2DFrozenArea[dst] = s.pCold2DFrozenArea[i];
    }
    }
    template<class State>
    static __device__ __forceinline__ void publish(State& s, const int dst)
    {
        if (s.compactPStuck[dst] != Foam::gpuThermal::particleWallMobile)
            Foam::gpuWall::publishWallBoundParticleIndex(s, dst);
    }
};
