#pragma once
template<class Real>
__device__ __forceinline__ float materialEnthalpyMoment(const Real mass, const Real temperature)
{
    return float(mass)*float(particleSpecificEnthalpyDevice(temperature));
}

// Material moments are derived from the single published thermal state.
template<bool CompactParticles = false, class State, class Real>
__device__ __forceinline__ float materialEnthalpyMomentFromParticle
(const State&, const int, const Real mass, const Real temperature, long)
{
    return materialEnthalpyMoment(mass, temperature);
}
