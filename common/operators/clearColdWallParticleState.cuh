#pragma once
#include <type_traits>
// One operator implementation; scalar/time adapters are compile-time only.
__device__ void clearColdWallParticleState
(
    DeviceState& s,
    const int particleI
)
{
    if (s.coldWallSolidificationEnabled == 0)
    {
        return;
    }
    const int nodeBase = particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase = particleI*Foam::gpuThermal::coldWallRadialRingCount;
    for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
    {
        s.pColdNodeSpecificEnthalpy[nodeBase + node] = 0.0;
    }
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0;
    }
    s.pColdFrozenArea[particleI] = 0.0;
    s.pColdContactAge[particleI] = 0.0;
}

__device__ void initialiseColdWallParticleState
(
    DeviceState& s,
    const int particleI,
    const GPU_OPERATOR_REAL temperatureK
)
{
    if
    (
        s.pColdNodeSpecificEnthalpy == nullptr
     || s.pColdRingSolidMass == nullptr
     || s.pColdFrozenArea == nullptr
     || s.pColdContactAge == nullptr
    )
    {
        asm("trap;");
    }
    GPU_OPERATOR_REAL nodeEnthalpy[Foam::gpuThermal::coldWallAxialNodeCount];
    GPU_OPERATOR_REAL ringSolidMass[Foam::gpuThermal::coldWallRadialRingCount];
    Foam::gpuThermal::initialiseColdWallProfile
    (
        nodeEnthalpy,
        ringSolidMass,
        temperatureK,
        s.coldWallSolidificationParameters
    );
    const int nodeBase = particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase = particleI*Foam::gpuThermal::coldWallRadialRingCount;
    for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
    {
        s.pColdNodeSpecificEnthalpy[nodeBase + node] =
            static_cast<typename std::remove_reference<decltype(s.pColdNodeSpecificEnthalpy[0])>::type>(nodeEnthalpy[node]);
    }
#ifdef GPU_PAIRED_PARTICLE_HEAT
    // The uniform float profile is a cache; retain its cast loss in the
    // existing physical J/kg carry without discarding earlier heat.
    s.pThermalEnthalpyResidual[particleI] += double(nodeEnthalpy[0])
      - double(s.pColdNodeSpecificEnthalpy[nodeBase]);
#endif
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0;
    }
    s.pColdFrozenArea[particleI] = 0.0;
    s.pColdContactAge[particleI] = 0.0;
}
