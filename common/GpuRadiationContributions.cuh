#pragma once
#include "GpuMaterialEnthalpyMoment.cuh"
struct ParticleRadiationContribution
{
    GPU_OPERATOR_REAL mass, temperatureMass, diameterMass;
};
__device__ __forceinline__ ParticleRadiationContribution particleRadiationContribution
(const DeviceState& s, const int i)
{
    const GPU_OPERATOR_REAL mass=clampMin(finiteOr(s.pm[i],GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL temperature=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
    const GPU_OPERATOR_REAL diameter=clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),GPU_OPERATOR_R(1.0e-12));
    return {mass,mass*temperature,mass*diameter};
}
__device__ __forceinline__ GPU_OPERATOR_REAL particleMaterialEnthalpyContribution
(const DeviceState& s, const int i)
{
    const GPU_OPERATOR_REAL mass=clampMin(finiteOr(s.pm[i],GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL temperature=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
    return materialEnthalpyMoment(mass, temperature);
}
