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
    if (!(physicalMass>0.) || !finiteThermalCarry(physicalMass)
        || !(parcelMass>=0.) || !finiteThermalCarry(parcelMass)
        || !(conductanceIntegral>=0.) || !finiteThermalCarry(wallTemperature)
        || !finiteThermalCarry(state.temperature) || !finiteThermalCarry(state.specificEnthalpyResidual))
        return {false,0.};
    const double oldEnthalpy = effectiveParticleEnthalpy(state, caloric);
    const double oldTemperature = caloric.temperature(oldEnthalpy);
    if (!finiteThermalCarry(oldEnthalpy) || !finiteThermalCarry(oldTemperature)) return {false,0.};
    if (conductanceIntegral==0.) return {true,0.};
    // Preserve the small enthalpy carry without subtracting absolute enthalpies.
    const double anchorCorrection=caloric.enthalpyIncrement(double(state.temperature),
        oldTemperature-double(state.temperature))-state.specificEnthalpyResidual;
    const double delta=wallTemperature-oldTemperature;
    const double equilibriumHeat=caloric.enthalpyIncrement(oldTemperature,delta)+anchorCorrection;
    const double cp=caloric.specificHeat(oldTemperature);
    if (!(cp>0.) || !finiteThermalCarry(cp) || !finiteThermalCarry(equilibriumHeat)) return {false,0.};
    const bool strong=conductanceIntegral/physicalMass>=cp;
    double proposedSpecificHeat=equilibriumHeat;
    if (finiteThermalCarry(conductanceIntegral)) {
        double lower=delta<0.?delta:0.,upper=delta>0.?delta:0.;
        for (int iteration=0;iteration<96;++iteration) {
            const double middle=lower+.5*(upper-lower);
            if(middle==lower||middle==upper)break;
            const double heat=caloric.enthalpyIncrement(oldTemperature,middle)+anchorCorrection;
            // Scale by the dominant coefficient; neither G*dT nor m*dh can overflow.
            const double residual=strong ? (physicalMass/conductanceIntegral)*heat+(middle-delta)
                : heat-(conductanceIntegral/physicalMass)*(delta-middle);
            if(!finiteThermalCarry(residual))return {false,0.};
            if(residual<0.)lower=middle;else upper=middle;
        }
        const double solvedDelta=lower+.5*(upper-lower);
        // Heat from enthalpy in the strong limit, from flux in the weak limit.
        proposedSpecificHeat=strong ? caloric.enthalpyIncrement(oldTemperature,solvedDelta)+anchorCorrection
            : (conductanceIntegral/physicalMass)*(delta-solvedDelta);
    }
    // Positive infinite conductance represents the zero-resistance ideal contact limit.
    auto candidate=state;
    const auto publication=publishParticleThermalHeatImpl<true>
        (candidate,proposedSpecificHeat,minimumTemperature,maximumTemperature,caloric);
    const double wallEnergy=-parcelMass*publication.acceptedSpecificHeat;
    if(!publication.valid||!finiteThermalCarry(wallEnergy))return {false,0.};
    state=candidate;
    return {true,wallEnergy};
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
