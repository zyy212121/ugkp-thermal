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
    float wallEnergy;
};

// The existing lumped implicit wall law, evaluated against effective material
// state. Publish through the same single rounded temperature representation as gas heat.
// Callers may work on a local candidate and commit after other checks succeed.
template<class Storage, class Caloric>
UGKWP_THERMAL_CONSUMER_HD inline ParticleWallThermalPublication publishFiniteWallThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const float wallTemperature,
    const float physicalMass,
    const float parcelMass,
    const float conductanceIntegral,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric,
    const int maximumIterations=96
)
{
    if (!(physicalMass>0.f) || !finiteThermalCarry(physicalMass)
        || !(parcelMass>=0.f) || !finiteThermalCarry(parcelMass)
        || !(conductanceIntegral>=0.f) || !finiteThermalCarry(wallTemperature)
        || !finiteThermalCarry(state.temperature))
        return {false,0.f};
    const float oldEnthalpy = effectiveParticleEnthalpy(state, caloric);
    const float oldTemperature = effectiveParticleTemperature(state,caloric);
    if (!finiteThermalCarry(oldEnthalpy) || !finiteThermalCarry(oldTemperature)) return {false,0.f};
    if (conductanceIntegral==0.f) return {true,0.f};

    const float delta=wallTemperature-oldTemperature;
    const float equilibriumHeat=caloric.enthalpyIncrement(oldTemperature,delta);
    const float cp=caloric.specificHeat(oldTemperature);
    if (!(cp>0.f) || !finiteThermalCarry(cp) || !finiteThermalCarry(equilibriumHeat)) return {false,0.f};
    const bool strong=conductanceIntegral/physicalMass>=cp;
    float proposedSpecificHeat=equilibriumHeat;
    if (finiteThermalCarry(conductanceIntegral)) {
        float lower=delta<0.f?delta:0.f,upper=delta>0.f?delta:0.f;
        for (int iteration=0;iteration<maximumIterations;++iteration) {
            const float middle=lower+.5f*(upper-lower);
            if(middle==lower||middle==upper)break;
            const float heat=caloric.enthalpyIncrement(oldTemperature,middle);
            // Scale by the dominant coefficient; neither G*dT nor m*dh can overflow.
            const float residual=strong ? (physicalMass/conductanceIntegral)*heat+(middle-delta)
                : heat-(conductanceIntegral/physicalMass)*(delta-middle);
            if(!finiteThermalCarry(residual))return {false,0.f};
            if(residual<0.f)lower=middle;else upper=middle;
        }
        const float solvedDelta=lower+.5f*(upper-lower);
        // Heat from enthalpy in the strong limit, from flux in the weak limit.
        proposedSpecificHeat=strong ? caloric.enthalpyIncrement(oldTemperature,solvedDelta)
            : (conductanceIntegral/physicalMass)*(delta-solvedDelta);
    }
    // Positive infinite conductance represents the zero-resistance ideal contact limit.
    auto candidate=state;
    const auto publication=publishParticleThermalHeatImpl<true>
        (candidate,proposedSpecificHeat,minimumTemperature,maximumTemperature,caloric);
    const float wallEnergy=-parcelMass*publication.acceptedSpecificHeat;
    if(!publication.valid||!finiteThermalCarry(wallEnergy))return {false,0.f};
    state=candidate;
    return {true,wallEnergy};
}

template<class Storage, class Caloric>
UGKWP_THERMAL_CONSUMER_HD inline ParticleThermalPublication publishAffineRadiationThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const float scale,
    const float offset,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    if (!finiteThermalCarry(scale) || !finiteThermalCarry(offset))
        return {false, false, 0.0f};
    if (scale == 1.0f && offset == 0.0f)
        return publishParticleThermalHeat(state, 0.0f, minimumTemperature, maximumTemperature, caloric);
    const float oldTemperature = effectiveParticleTemperature(state, caloric);
    const float temperatureIncrement = ::fma(scale - 1.0f, oldTemperature, offset);
    const float proposedTemperature = oldTemperature + temperatureIncrement;
    if (!finiteThermalCarry(oldTemperature) || !finiteThermalCarry(proposedTemperature))
        return {false, false, 0.0f};
    // Integrate the small interval before adding it to the large thermal state.
    const float proposedSpecificHeat = caloric.enthalpyIncrement(oldTemperature, temperatureIncrement);


    if (scale >= 0.0f && scale <= 1.0f
        && proposedTemperature >= float(minimumTemperature)
        && proposedTemperature <= float(maximumTemperature))
        return publishParticleThermalHeat
            (state, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
    return publishParticleThermalHeatImpl<false>
        (state, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
}
}
#undef UGKWP_THERMAL_CONSUMER_HD
