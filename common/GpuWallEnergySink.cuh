#pragma once
template<class State>
__device__ __forceinline__ auto publishLimitedGasWallEnergy(State& s, const int f, int)
    -> decltype(s.wallEnergy.gasWallEnergy, void())
{
    if (s.wallEnergy.gasWallEnergy != nullptr
        && s.wallEnergy.gasWallEnergyMask != nullptr
        && s.wallEnergy.gasWallEnergyMask[f] != 0
        && f >= s.nInternalFaces)
        s.wallEnergy.gasWallFlux[f] = s.gasBoundaryKind[f] == 2
            ? static_cast<double>(s.gasPhiRhoE[f]) : 0.0;
}
template<class State>
__device__ __forceinline__ void publishLimitedGasWallEnergy(State&, const int, long) {}
