#pragma once
// CHT owns this optional persistent field. Other resident states have no
// allocation or extra payload; overload selection happens at compile time.
template<class State>
__device__ __forceinline__ auto resetParticleThermalEnthalpyResidual(State& s, const int i, int)
    -> decltype(s.pThermalEnthalpyResidual[i] = 0.0, void())
{
    s.pThermalEnthalpyResidual[i] = 0.0;
}
template<class State>
__device__ __forceinline__ void resetParticleThermalEnthalpyResidual(State&, const int, long) {}

template<class State>
__device__ __forceinline__ auto copyParticleThermalEnthalpyResidual
(State& s, const int i, const int dst, int)
    -> decltype(s.compactPThermalEnthalpyResidual[dst] = s.pThermalEnthalpyResidual[i], void())
{
    s.compactPThermalEnthalpyResidual[dst] = s.pThermalEnthalpyResidual[i];
}
template<class State>
__device__ __forceinline__ void copyParticleThermalEnthalpyResidual(State&, const int, const int, long) {}

template<class State>
__host__ __device__ __forceinline__ auto swapParticleThermalEnthalpyResidual(State& s, int)
    -> decltype(s.pThermalEnthalpyResidual = s.compactPThermalEnthalpyResidual, void())
{
    auto* previous = s.pThermalEnthalpyResidual;
    s.pThermalEnthalpyResidual = s.compactPThermalEnthalpyResidual;
    s.compactPThermalEnthalpyResidual = previous;
}
template<class State>
__host__ __device__ __forceinline__ void swapParticleThermalEnthalpyResidual(State&, long) {}
