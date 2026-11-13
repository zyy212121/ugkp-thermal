#pragma once
#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define UGKWP_THERMAL_CARRY_HD __host__ __device__
#else
#define UGKWP_THERMAL_CARRY_HD
#endif

namespace ugkwp
{
// FP32 thermal authority is one stored temperature. There is no rounding carry.
template<class TemperatureStorage>
struct ParticleThermalCarry
{
    TemperatureStorage temperature;
};
struct ParticleThermalPublication
{
    bool valid;
    bool limited;
    float acceptedSpecificHeat;
};
UGKWP_THERMAL_CARRY_HD inline bool finiteThermalCarry(const float x)
{
    return x==x && x<=3.4028234663852886e38f && x>=-3.4028234663852886e38f;
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline float effectiveParticleEnthalpy
(const ParticleThermalCarry<Storage>& state,const Caloric& caloric)
{
    return caloric.enthalpy(float(state.temperature));
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline float effectiveParticleTemperature
(const ParticleThermalCarry<Storage>& state,const Caloric& caloric)
{
    (void)caloric;
    return float(state.temperature);
}

// Compute a candidate before changing the single thermal state. Caloric
// enthalpy(T) and temperature(h) operations both use binary32.
// The caller pairs acceptedSpecificHeat with its other phase and commits only


template<bool LimitAcceptedHeat, bool TerminalPublication = false, class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline ParticleThermalPublication publishParticleThermalHeatImpl
(
    ParticleThermalCarry<Storage>& state,
    const float proposedSpecificHeat,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    const float base=caloric.enthalpy(float(state.temperature));

    if (!TerminalPublication && (!finiteThermalCarry(base)
        ||!finiteThermalCarry(proposedSpecificHeat)))
        return {false,false,0.0f};
    if (proposedSpecificHeat==0.0f) return {true,false,0.0f};

    float accepted=proposedSpecificHeat;
    if constexpr (LimitAcceptedHeat)
    {


        const float minimum=caloric.enthalpy(float(minimumTemperature));
        const float maximum=caloric.enthalpy(float(maximumTemperature));
        if (!TerminalPublication && (!finiteThermalCarry(minimum)||!finiteThermalCarry(maximum)))
            return {false,false,0.0f};
        const float lowerBound=minimum-base;
        const float upperBound=maximum-base;
        const bool originalState = minimum<=maximum
            && base>=minimum && base<=maximum;
        const float lower=originalState ? lowerBound : (lowerBound<0.0f?lowerBound:0.0f);
        const float upper=originalState ? upperBound : (upperBound>0.0f?upperBound:0.0f);
        accepted=proposedSpecificHeat<lower ? lower
            : (proposedSpecificHeat>upper ? upper : proposedSpecificHeat);
    }
    const float target=base+accepted;
    const float candidateTemperature=caloric.temperature(target);
    if (!TerminalPublication && (!finiteThermalCarry(candidateTemperature))) return {false,false,0.0f};
    Storage published=Storage(candidateTemperature);
    if (published<minimumTemperature) published=minimumTemperature;
    if (published>maximumTemperature) published=maximumTemperature;
    const float publishedBase=caloric.enthalpy(float(published));
    // Accepted exchange is measured from the single rounded published state.
    const float actual=publishedBase-base;
    if (!TerminalPublication && (!finiteThermalCarry(publishedBase)
        ||!finiteThermalCarry(actual))) return {false,false,0.0f};
    state.temperature=published;
    return {finiteThermalCarry(actual),accepted!=proposedSpecificHeat,actual};
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline ParticleThermalPublication publishParticleThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const float proposedSpecificHeat,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    return publishParticleThermalHeatImpl<true>
        (state,proposedSpecificHeat,minimumTemperature,maximumTemperature,caloric);
}
}
#undef UGKWP_THERMAL_CARRY_HD
