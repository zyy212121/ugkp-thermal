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
// The residual is specific enthalpy [J/kg], never a temperature or an intended
// exchange counter. Both components together are the represented thermal state.
template<class TemperatureStorage>
struct ParticleThermalCarry
{
    TemperatureStorage temperature;
    double specificEnthalpyResidual;
};
struct ParticleThermalPublication
{
    bool valid;
    bool limited;
    double acceptedSpecificHeat;
};
UGKWP_THERMAL_CARRY_HD inline bool finiteThermalCarry(const double x)
{
    return x==x && x<=1.7976931348623157e308 && x>=-1.7976931348623157e308;
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline double effectiveParticleEnthalpy
(const ParticleThermalCarry<Storage>& state,const Caloric& caloric)
{
    return caloric.enthalpy(double(state.temperature))+state.specificEnthalpyResidual;
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline double effectiveParticleTemperature
(const ParticleThermalCarry<Storage>& state,const Caloric& caloric)
{
    return caloric.temperature(effectiveParticleEnthalpy(state,caloric));
}

// Compute a candidate completely before changing either component. Caloric must
// provide mutually consistent double enthalpy(T) and temperature(h) operations.
// The caller pairs acceptedSpecificHeat with its other phase and commits only
// after validating the whole shared transaction. No production trap is added.
template<bool LimitAcceptedHeat, class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline ParticleThermalPublication publishParticleThermalHeatImpl
(
    ParticleThermalCarry<Storage>& state,
    const double proposedSpecificHeat,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    const double base=caloric.enthalpy(double(state.temperature));
    const double residual=state.specificEnthalpyResidual;
    if (!finiteThermalCarry(base)||!finiteThermalCarry(residual)
        ||!finiteThermalCarry(proposedSpecificHeat))
        return {false,false,0.0};
    if (proposedSpecificHeat==0.0) return {true,false,0.0};

    double accepted=proposedSpecificHeat;
    if constexpr (LimitAcceptedHeat)
    {
        // Preserve the original endpoint cancellation on admitted states.
        // Only formerly refused represented states use the extension interval.
        const double minimum=caloric.enthalpy(double(minimumTemperature));
        const double maximum=caloric.enthalpy(double(maximumTemperature));
        if (!finiteThermalCarry(minimum)||!finiteThermalCarry(maximum))
            return {false,false,0.0};
        const double lowerBound=(minimum-base)-residual;
        const double upperBound=(maximum-base)-residual;
        const bool originalState = minimum<=maximum
            && base+residual>=minimum && base+residual<=maximum;
        const double lower=originalState ? lowerBound : (lowerBound<0.0?lowerBound:0.0);
        const double upper=originalState ? upperBound : (upperBound>0.0?upperBound:0.0);
        accepted=proposedSpecificHeat<lower ? lower
            : (proposedSpecificHeat>upper ? upper : proposedSpecificHeat);
    }
    const double target=base+(residual+accepted);
    const double candidateTemperature=caloric.temperature(target);
    if (!finiteThermalCarry(candidateTemperature)) return {false,false,0.0};
    Storage published=Storage(candidateTemperature);
    if (published<minimumTemperature) published=minimumTemperature;
    if (published>maximumTemperature) published=maximumTemperature;
    const double publishedBase=caloric.enthalpy(double(published));
    // Do not form target-publishedBase: that would discard sub-double-ULP heat
    // when it was first added to the large base. Keep the small component local.
    const double publishedResidual=((base-publishedBase)+residual)+accepted;
    const double actual=(publishedBase-base)+(publishedResidual-residual);
    if (!finiteThermalCarry(publishedBase)||!finiteThermalCarry(publishedResidual)
        ||!finiteThermalCarry(actual)) return {false,false,0.0};
    state.temperature=published;
    state.specificEnthalpyResidual=publishedResidual;
    return {true,accepted!=proposedSpecificHeat,actual};
}

template<class Storage,class Caloric>
UGKWP_THERMAL_CARRY_HD inline ParticleThermalPublication publishParticleThermalHeat
(
    ParticleThermalCarry<Storage>& state,
    const double proposedSpecificHeat,
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
