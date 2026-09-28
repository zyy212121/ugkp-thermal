#pragma once
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
        s.pColdNodeSpecificEnthalpy[nodeBase + node] = 0.0f;
    }
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0f;
    }
    s.pColdFrozenArea[particleI] = 0.0f;
    s.pColdContactAge[particleI] = 0.0f;
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
        s.coldWallSolidificationEnabled == 0
     || s.pColdNodeSpecificEnthalpy == nullptr
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
            static_cast<float>(nodeEnthalpy[node]);
    }
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0f;
    }
    s.pColdFrozenArea[particleI] = 0.0f;
    s.pColdContactAge[particleI] = 0.0f;
}
