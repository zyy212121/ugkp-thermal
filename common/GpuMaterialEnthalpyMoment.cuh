#pragma once
template<class Real>
__device__ __forceinline__ Real materialEnthalpyMoment(const Real mass, const Real temperature)
{
    return mass*particleSpecificEnthalpyDevice(temperature);
}




template<bool CompactParticles = false, class State, class Real>
__device__ __forceinline__ auto materialEnthalpyMomentFromParticle
(const State& s, const int i, const Real mass, const Real temperature, int)
    -> decltype(s.pThermalEnthalpyResidual[i], double(0))
{
    return double(materialEnthalpyMoment(mass, temperature)) + double(mass)*
        (CompactParticles ? s.compactPThermalEnthalpyResidual[i] : s.pThermalEnthalpyResidual[i]);
}

template<bool CompactParticles = false, class State, class Real>
__device__ __forceinline__ Real materialEnthalpyMomentFromParticle
(const State&, const int, const Real mass, const Real temperature, long)
{
    return materialEnthalpyMoment(mass, temperature);
}
