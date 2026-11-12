#pragma once
#include "GpuMaterialEnthalpyMoment.cuh"
struct ParticleRadiationContribution
{
    GpuEnergy mass, temperatureMass, diameterMass;
};
__device__ __forceinline__ ParticleRadiationContribution particleRadiationContribution
(const DeviceState& s, const int i)
{
    const GpuEnergy mass=clampMin(finiteOr(s.pm[i],GpuEnergy(0.0)),GpuEnergy(0.0));
    const GpuEnergy temperature=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
    const GpuEnergy diameter=clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),GpuEnergy(1.0e-12));
    return {mass,mass*temperature,mass*diameter};
}
__device__ __forceinline__ GpuEnergy particleMaterialEnthalpyContribution
(const DeviceState& s, const int i)
{
    const GpuEnergy mass=clampMin(finiteOr(s.pm[i],GpuEnergy(0.0)),GpuEnergy(0.0));
    const GpuEnergy temperature=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
    return materialEnthalpyMomentFromParticle(s, i, mass, temperature, 0);
}
