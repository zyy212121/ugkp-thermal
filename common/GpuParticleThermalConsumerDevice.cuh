#pragma once
#include "GpuParticleThermalConsumers.H"
#include <type_traits>

struct ParticleFiniteContactThermalCandidate
{
    bool valid;
    double particleTemperatureK;
    double specificEnthalpyResidual;
    double wallEnergyJ;
};

template<class State, class Real>
__device__ __forceinline__ auto particleFiniteContactThermalCandidateDevice
(const State& s, const int i, const Real wallTemperature, const Real physicalMass,
 const Real parcelMass, const Real conductanceIntegral, int)
    -> decltype(s.pThermalEnthalpyResidual[i], ParticleFiniteContactThermalCandidate{})
{
    using Storage = typename std::decay<decltype(s.pT[i])>::type;
    ugkwp::ParticleThermalCarry<Storage> candidate
        {s.pT[i], s.pThermalEnthalpyResidual[i]};
    const auto publication = ugkwp::publishFiniteWallThermalHeat
    (
        candidate, double(wallTemperature), double(physicalMass), double(parcelMass),
        double(conductanceIntegral), Storage(s.TpMin), Storage(s.TpMax),
        particleCaloricDevice(s, i)
    );
    return {publication.valid, double(candidate.temperature),
        candidate.specificEnthalpyResidual, publication.wallEnergy};
}
template<class State, class Real>
__device__ __forceinline__ ParticleFiniteContactThermalCandidate particleFiniteContactThermalCandidateDevice
(const State& s, const int i, const Real wallTemperature, const Real physicalMass,
 const Real parcelMass, const Real conductanceIntegral, long)
{
    const auto result = Foam::gpuThermal::lumpedParticleWallInterfaceContact
        (s.pT[i], wallTemperature, physicalMass, parcelMass, conductanceIntegral);
    return {result.valid, double(result.particleTemperatureK), 0.0, double(result.wallEnergyJ)};
}

template<class State>
__device__ __forceinline__ auto commitParticleFiniteContactThermalCandidateDevice
(State& s, const int i, const ParticleFiniteContactThermalCandidate& candidate, int)
    -> decltype(s.pThermalEnthalpyResidual[i] = candidate.specificEnthalpyResidual, void())
{
    if(!candidate.valid)return;
    s.pT[i] = candidate.particleTemperatureK;
    s.pThermalEnthalpyResidual[i] = candidate.specificEnthalpyResidual;
}
template<class State>
__device__ __forceinline__ void commitParticleFiniteContactThermalCandidateDevice
(State& s, const int i, const ParticleFiniteContactThermalCandidate& candidate, long)
{
    if(!candidate.valid)return;
    s.pT[i] = candidate.particleTemperatureK;
}
