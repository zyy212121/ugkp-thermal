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
    const Caloric& caloric,
    const int maximumIterations=96
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
        for (int iteration=0;iteration<maximumIterations;++iteration) {
            const double middle=lower+.5*(upper-lower);
            if(middle==lower||middle==upper)break;
            const double heat=caloric.enthalpyIncrement(oldTemperature,middle)+anchorCorrection;

            const double residual=strong ? (physicalMass/conductanceIntegral)*heat+(middle-delta)
                : heat-(conductanceIntegral/physicalMass)*(delta-middle);
            if(!finiteThermalCarry(residual))return {false,0.};
            if(residual<0.)lower=middle;else upper=middle;
        }
        const double solvedDelta=lower+.5*(upper-lower);

        proposedSpecificHeat=strong ? caloric.enthalpyIncrement(oldTemperature,solvedDelta)+anchorCorrection
            : (conductanceIntegral/physicalMass)*(delta-solvedDelta);
    }

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
