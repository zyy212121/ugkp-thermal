#pragma once
#include "GpuParticleThermalCarry.H"

#if defined(__CUDACC__)
#define UGKWP_THERMAL_CONSUMER_HD __host__ __device__
#else
#define UGKWP_THERMAL_CONSUMER_HD
#endif
namespace ugkwp
{
struct ParticleWallThermalPublication
{
    bool valid;
    double wallEnergy;
};

// The existing lumped implicit wall law, evaluated against effective material
// state. Publish through the same bounded carry representation as gas heat.
// Callers may work on a local candidate and commit after other checks succeed.
template<class Storage, class Caloric>
UGKWP_THERMAL_CONSUMER_HD inline ParticleWallThermalPublication publishFiniteWallThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const double wallTemperature,
    const double physicalMass,
    const double parcelMass,
    const double conductanceIntegral,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    const double oldEnthalpy = effectiveParticleEnthalpy(state, caloric);
    const double oldTemperature = caloric.temperature(oldEnthalpy);
    if (conductanceIntegral == 0.0 || wallTemperature == oldTemperature)
    {
        const auto publication = publishParticleThermalHeatImpl<true, true>
            (state, 0.0, minimumTemperature, maximumTemperature, caloric);
        return {publication.valid, 0.0};
    }
    double lower = oldTemperature < wallTemperature ? oldTemperature : wallTemperature;
    double upper = oldTemperature > wallTemperature ? oldTemperature : wallTemperature;
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        const double middle = lower + 0.5*(upper - lower);
        const double residual = physicalMass*(caloric.enthalpy(middle) - oldEnthalpy)
            + conductanceIntegral*(middle - wallTemperature);
        if (residual < 0.0) lower = middle;
        else upper = middle;
        if (lower + 0.5*(upper - lower) == middle) break;
    }
    const double solvedTemperature = lower + 0.5*(upper - lower);
    // The small flux stays small; subtracting two large absolute h values here
    // would discard heat that the persistent carry is meant to retain.
    const double proposedSpecificHeat =
        conductanceIntegral*(wallTemperature - solvedTemperature)/physicalMass;
    auto candidate = state;
    const auto publication = publishParticleThermalHeatImpl<true, true>
        (candidate, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
    const double wallEnergy = -parcelMass*publication.acceptedSpecificHeat;
    state = candidate;
    return {publication.valid && finiteThermalCarry(wallEnergy), wallEnergy};
}

template<class Storage, class Caloric>
UGKWP_THERMAL_CONSUMER_HD inline ParticleThermalPublication publishAffineRadiationThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const double scale,
    const double offset,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    if (!finiteThermalCarry(scale) || !finiteThermalCarry(offset))
        return {false, false, 0.0};
    if (scale == 1.0 && offset == 0.0)
        return publishParticleThermalHeat(state, 0.0, minimumTemperature, maximumTemperature, caloric);
    const double oldTemperature = effectiveParticleTemperature(state, caloric);
    const double temperatureIncrement = ::fma(scale - 1.0, oldTemperature, offset);
    const double proposedTemperature = oldTemperature + temperatureIncrement;
    if (!finiteThermalCarry(oldTemperature) || !finiteThermalCarry(proposedTemperature))
        return {false, false, 0.0};
    // Integrate the small interval before adding it to the large thermal state.
    const double proposedSpecificHeat = caloric.enthalpyIncrement(oldTemperature, temperatureIncrement);
    // The old accepted affine transaction used the ordinary heat limiter,
    // including endpoint roundoff. Only old-refused transforms are unlimited.
    if (scale >= 0.0 && scale <= 1.0
        && proposedTemperature >= double(minimumTemperature)
        && proposedTemperature <= double(maximumTemperature))
        return publishParticleThermalHeat
            (state, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
    return publishParticleThermalHeatImpl<false>
        (state, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
}
}
#undef UGKWP_THERMAL_CONSUMER_HD
