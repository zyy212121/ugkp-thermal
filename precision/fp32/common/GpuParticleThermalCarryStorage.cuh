#pragma once
// FP32 has no secondary enthalpy storage to reset, compact or swap.
template<class State> __device__ __forceinline__ void resetParticleThermalEnthalpyResidual(State&, int, int) {}
template<class State> __device__ __forceinline__ void copyParticleThermalEnthalpyResidual(State&, int, int, int) {}
template<class State> __host__ __device__ __forceinline__ void swapParticleThermalEnthalpyResidual(State&, int) {}
