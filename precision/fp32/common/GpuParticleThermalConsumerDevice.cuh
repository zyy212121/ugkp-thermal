#pragma once
#include "GpuParticleThermalConsumers.H"
#include <type_traits>

struct ParticleFiniteContactThermalCandidate
{
    bool valid;
    float particleTemperatureK;
    float wallEnergyJ;
};

#ifdef GPU_PAIRED_PARTICLE_HEAT
template<class State, class Real>
__device__ __forceinline__ auto particleFiniteContactThermalCandidateDevice
(const State& s, const int i, const Real wallTemperature, const Real physicalMass,
 const Real parcelMass, const Real conductanceIntegral, int)
    -> decltype(s.pT[i], ParticleFiniteContactThermalCandidate{})
{
    using Storage = typename std::decay<decltype(s.pT[i])>::type;
    ugkwp::ParticleThermalCarry<Storage> candidate
        {s.pT[i]};
    const auto publication = ugkwp::publishFiniteWallThermalHeat
    (
        candidate, float(wallTemperature), float(physicalMass), float(parcelMass),
        float(conductanceIntegral), Storage(s.TpMin), Storage(s.TpMax),
        particleCaloricDevice(s, i),s.particleFiniteContactMaximumIterations
    );
    return {publication.valid, float(candidate.temperature), publication.wallEnergy};
}
#endif
template<class State, class Real>
__device__ __forceinline__ ParticleFiniteContactThermalCandidate particleFiniteContactThermalCandidateDevice
(const State& s, const int i, const Real wallTemperature, const Real physicalMass,
 const Real parcelMass, const Real conductanceIntegral, long)
{
    const auto result = Foam::gpuThermal::lumpedParticleWallInterfaceContact
        (s.pT[i], wallTemperature, physicalMass, parcelMass, conductanceIntegral);
    return {result.valid, float(result.particleTemperatureK), float(result.wallEnergyJ)};
}

template<class State>
__device__ __forceinline__ auto commitParticleFiniteContactThermalCandidateDevice
(State& s, const int i, const ParticleFiniteContactThermalCandidate& candidate, int)
    -> decltype(s.pT[i] = candidate.particleTemperatureK, void())
{
    if(!candidate.valid)return;
    s.pT[i] = candidate.particleTemperatureK;
}
template<class State>
__device__ __forceinline__ void commitParticleFiniteContactThermalCandidateDevice
(State& s, const int i, const ParticleFiniteContactThermalCandidate& candidate, long)
{
    if(!candidate.valid)return;
    s.pT[i] = candidate.particleTemperatureK;
}
