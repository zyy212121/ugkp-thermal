#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_FINITE_WALL_CONTACT_H
#define GPU_THERMAL_GPU_FINITE_WALL_CONTACT_H

#include "GpuAluminaProperties.H"
#include "GpuWallInterfaceHeat.H"

#include <cmath>
#include <limits>
#include <cfloat>

#if defined(__CUDACC__)
#define GPU_FINITE_CONTACT_HD __host__ __device__
#else
#define GPU_FINITE_CONTACT_HD
#endif

namespace Foam
{
namespace gpuThermal
{
constexpr GpuReal finiteContactPi =
    GPU_R(3.141592653589793238462643383279502884);
constexpr int finiteContactTimeTableSize = 257;
constexpr int finiteContactDamageTableSize = 65;
constexpr int finiteContactPeakTableSize = 33;
constexpr int finiteContactThermalRingCount = 8;

                                                                           
                                                                              
constexpr GpuReal defaultWallDensityKgM3 = GPU_R(1800.0);
constexpr GpuReal defaultWallSpecificHeatJkgK = GPU_R(710.0);
constexpr GpuReal defaultWallConductivityWmK = GPU_R(100.0);

enum ParticleWallState : unsigned char
{
    particleWallMobile = 0,
    particleWallDeposited = 1,
    particleWallTransientRebound = 2,
    particleWallTransientDeposit = 3
};

struct FiniteWallContactImpact
{
    bool valid;
    GpuReal betaMax;
    GpuReal maximumAreaM2;
    GpuTime contactDurationS;
    GpuReal peakTimeFraction;
    GpuReal physicalMassKg;
    GpuReal weber;
    GpuReal ohnesorge;
};

template<class Real>
GPU_FINITE_CONTACT_HD inline bool finiteContactValue(const Real x) noexcept
{
    return x == x && x >= -std::numeric_limits<Real>::max()
        && x <= std::numeric_limits<Real>::max();
}

GPU_FINITE_CONTACT_HD inline GpuReal clampFiniteContact
(
    const GpuReal x,
    const GpuReal lo,
    const GpuReal hi
) noexcept
{
    return x < lo ? lo : (x > hi ? hi : x);
}

GPU_FINITE_CONTACT_HD inline GpuReal finiteContactPhi
(
    const GpuReal beta,
    const GpuReal oneMinusCosContactAngle
) noexcept
{
    return
        GPU_R(2.0)/(GPU_R(3.0)*beta)
      + GPU_R(0.25)*beta*beta*oneMinusCosContactAngle
      - GPU_R(1.0);
}

GPU_FINITE_CONTACT_HD inline GpuReal correctedSpreadResidual
(
    const GpuReal beta,
    const GpuReal weber,
    const GpuReal ohnesorge,
    const GpuReal oneMinusCosContactAngle
) noexcept
{
    const GpuReal ratio = beta/GPU_R(0.61);
    const GpuReal ratio2 = ratio*ratio;
    const GpuReal ratio6 = ratio2*ratio2*ratio2;
    return
        ohnesorge*(ratio6 - GPU_R(23.3))
      - weber
      + GPU_R(12.0)*finiteContactPhi(beta, oneMinusCosContactAngle);
}

GPU_FINITE_CONTACT_HD inline GpuReal correctedMaximumSpreadFactor
(
    const GpuReal weber,
    const GpuReal ohnesorge,
    const GpuReal contactAngleCosine
) noexcept
{
    if (!finiteContactValue(weber) || !finiteContactValue(ohnesorge)
        || !finiteContactValue(contactAngleCosine))
    {
        return std::numeric_limits<GpuReal>::quiet_NaN();
    }
    const GpuReal oneMinusCosContactAngle = GPU_R(1.0) - contactAngleCosine;

    const GpuReal uncorrected =
        GPU_R(0.61)*::pow(GPU_R(23.3) + weber/ohnesorge, GPU_R(1.0)/GPU_R(6.0));
    if (!finiteContactValue(uncorrected))
    {
        return std::numeric_limits<GpuReal>::quiet_NaN();
    }

                                                                         
                                                                     
                                                                           
                                                                         
                                                                            
    if (uncorrected <= GPU_R(1.0))
    {
        return GPU_R(1.0);
    }

    GpuReal lower = GPU_R(1.0);
    GpuReal upper = uncorrected;
    GpuReal lowerResidual = correctedSpreadResidual
    (
        lower, weber, ohnesorge, oneMinusCosContactAngle
    );
    GpuReal upperResidual = correctedSpreadResidual
    (
        upper, weber, ohnesorge, oneMinusCosContactAngle
    );
    if
    (
        !finiteContactValue(lowerResidual)
     || !finiteContactValue(upperResidual)
    )
    {
        return std::numeric_limits<GpuReal>::quiet_NaN();
    }
    if (lowerResidual >= GPU_R(0.0))
    {
        return GPU_R(1.0);
    }

    int expansion = 0;
    while (!(upperResidual > GPU_R(0.0)) && expansion < 40)
    {
        lower = upper;
        lowerResidual = upperResidual;
        upper *= GPU_R(2.0);
        if (!finiteContactValue(upper))
        {
            return std::numeric_limits<GpuReal>::quiet_NaN();
        }
        upperResidual = correctedSpreadResidual
        (
            upper, weber, ohnesorge, oneMinusCosContactAngle
        );
        if (!finiteContactValue(upperResidual))
        {
            return std::numeric_limits<GpuReal>::quiet_NaN();
        }
        ++expansion;
    }
    if (!(upperResidual > GPU_R(0.0)))
    {
        return upper;
    }

    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const GpuReal middle = GPU_R(0.5)*(lower + upper);
        const GpuReal middleResidual = correctedSpreadResidual
        (
            middle, weber, ohnesorge, oneMinusCosContactAngle
        );
        if (!finiteContactValue(middleResidual))
        {
            return std::numeric_limits<GpuReal>::quiet_NaN();
        }
        if (middleResidual <= GPU_R(0.0))
        {
            lower = middle;
            lowerResidual = middleResidual;
        }
        else
        {
            upper = middle;
        }
    }
    const GpuReal beta = GPU_R(0.5)*(lower + upper);
    return beta;
}

GPU_FINITE_CONTACT_HD inline FiniteWallContactImpact evaluateFiniteWallContactImpact
(
    const GpuReal particleTemperatureK,
    const GpuReal diameterM,
    const GpuReal normalSpeedMps,
    const GpuReal contactAngleCosine
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);

    const GpuReal weber =
        material.densityKgM3*normalSpeedMps*normalSpeedMps*diameterM
       /material.surfaceTensionNM;
    const GpuReal ohnesorge =
        material.viscosityPaS
       /::sqrt(material.densityKgM3*material.surfaceTensionNM*diameterM);
    const GpuReal beta = correctedMaximumSpreadFactor
    (
        weber, ohnesorge, contactAngleCosine
    );
    const GpuTime spreadingDuration =
        GPU_R(0.5)*beta*diameterM/normalSpeedMps;
    const GpuTime capillaryTime = ::sqrt
    (
        material.densityKgM3*diameterM*diameterM*diameterM
       /material.surfaceTensionNM
    );
    const GpuReal naturalFrequency = GPU_R(8.0)/capillaryTime;
    const GpuReal dampingRate =
        GPU_R(20.0)*material.viscosityPaS
       /(material.densityKgM3*diameterM*diameterM);
    const GpuReal dampedFrequencySquared =
        naturalFrequency*naturalFrequency - dampingRate*dampingRate;

    const GpuReal dampedFrequency = ::sqrt(dampedFrequencySquared);
    const GpuTime recoilDuration =
        (finiteContactPi - ::atan2(dampedFrequency, dampingRate))
       /dampedFrequency;
    const GpuTime duration = spreadingDuration + recoilDuration;
    GpuReal peakTimeFraction = static_cast<GpuReal>(spreadingDuration/duration);
    const GpuReal maximumArea =
        (finiteContactPi/GPU_R(4.0))*beta*beta*diameterM*diameterM;
    const GpuReal physicalMass =
        (finiteContactPi/GPU_R(6.0))*material.densityKgM3
       *diameterM*diameterM*diameterM;
    const bool valid =
        finiteContactValue(beta) && finiteContactValue(spreadingDuration)
     && finiteContactValue(recoilDuration) && finiteContactValue(duration)
     && finiteContactValue(physicalMass) && finiteContactValue(peakTimeFraction)
     && finiteContactValue(maximumArea);
    return
    {
        valid,
        beta,
        maximumArea,
        duration,
        peakTimeFraction,
        physicalMass,
        weber,
        ohnesorge
    };
}



GPU_FINITE_CONTACT_HD inline GpuReal normalizedKinematicArea
(
    const GpuReal normalizedAge,
    const GpuReal peakTimeFraction
) noexcept
{
    if (!(normalizedAge > GPU_R(0.0)) || normalizedAge >= GPU_R(1.0))
    {
        return GPU_R(0.0);
    }
    const GpuReal theta = clampFiniteContact(normalizedAge, GPU_R(0.0), GPU_R(1.0));
    const GpuReal peak = clampFiniteContact(peakTimeFraction, GPU_R(0.0), GPU_R(1.0));
    if (!(peak > GPU_R(0.0)))
    {
        const GpuReal cosine = ::cos(GPU_R(0.5)*finiteContactPi*theta);
        return cosine*cosine;
    }
    if (!(peak < GPU_R(1.0)))
    {
        return theta*(GPU_R(2.0) - theta);
    }
    if (theta <= peak)
    {
        const GpuReal x = theta/peak;
        return x*(GPU_R(2.0) - x);
    }
    const GpuReal y = (theta - peak)/(GPU_R(1.0) - peak);
    const GpuReal cosine = ::cos(GPU_R(0.5)*finiteContactPi*y);
    return cosine*cosine;
}

GPU_FINITE_CONTACT_HD inline GpuReal normalizedEffectiveArea
(
    const GpuReal normalizedAge,
    const GpuReal peakTimeFraction,
    const GpuReal normalizedDamage
) noexcept
{
    const GpuReal value =
        normalizedKinematicArea(normalizedAge, peakTimeFraction)
      - clampFiniteContact(normalizedDamage, GPU_R(0.0), GPU_R(1.0));
    return value > GPU_R(0.0) ? value : GPU_R(0.0);
}

GPU_FINITE_CONTACT_HD inline GpuReal finiteContactWallConductanceTimeIntegral
(
    const GpuReal maximumAreaM2,
    const GpuReal contactAreaM2,
    const GpuTime contactDurationS,
    const GpuReal peakTimeFraction,
    const GpuTime profileAge0S,
    const GpuTime deltaTSeconds,
    const GpuReal efficiency,
    const GpuReal interfaceResistanceM2KW,
    const GpuReal particleAdditionalResistanceM2KW,
    const GpuReal wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if (!(contactAreaM2 > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    if (deltaTSeconds == GPU_R(0.0))
    {
        return GPU_R(0.0);
    }
    const GpuReal ringArea =
        maximumAreaM2/static_cast<GpuReal>(finiteContactThermalRingCount);
    GpuReal total = GPU_R(0.0);
    for (int ring = 0; ring < finiteContactThermalRingCount; ++ring)
    {
        const GpuReal ringInnerArea = static_cast<GpuReal>(ring)*ringArea;
        const GpuReal ringWetArea = clampFiniteContact
        (
            contactAreaM2 - ringInnerArea,
            GPU_R(0.0),
            ringArea
        );
        if (!(ringWetArea > GPU_R(0.0)))
        {
            continue;
        }
        const GpuReal radialFraction =
            (static_cast<GpuReal>(ring) + GPU_R(0.5))
           /static_cast<GpuReal>(finiteContactThermalRingCount);
        const GpuReal spreadCoordinate =
            GPU_R(1.0) - ::sqrt(clampFiniteContact(GPU_R(1.0) - radialFraction, GPU_R(0.0), GPU_R(1.0)));
        const GpuTime firstWetAgeS =
            contactDurationS*peakTimeFraction*spreadCoordinate;
        const GpuTime localAge0S = profileAge0S > firstWetAgeS
          ? profileAge0S - firstWetAgeS
          : GPU_R(0.0);
        const GpuTime profileAge1S = profileAge0S + deltaTSeconds;
        const GpuTime localAge1S = profileAge1S > firstWetAgeS
          ? profileAge1S - firstWetAgeS
          : GPU_R(0.0);
        const GpuReal contribution = wallInterfaceConductanceTimeIntegral
        (
            ringWetArea,
            efficiency,
            localAge0S,
            localAge1S,
            interfaceResistanceM2KW,
            particleAdditionalResistanceM2KW,
            wallEffusivityWsHalfM2K,
            wallTransientResistance
        );

        total += contribution;
    }
    return total;
}

GPU_FINITE_CONTACT_HD inline GpuReal combinedContactEffusivityFromWallEffusivity
(
    const GpuReal particleTemperatureK,
    const GpuReal wallEffusivity
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if (!material.valid || !(wallEffusivity > GPU_R(0.0))
     || !finiteContactValue(wallEffusivity))
    {
        return -GPU_R(1.0);
    }
    const GpuReal particle = ::sqrt
    (
        material.thermalConductivityWmK*material.densityKgM3
       *material.specificHeatJkgK
    );
    return particle*wallEffusivity/(particle + wallEffusivity);
}

GPU_FINITE_CONTACT_HD inline GpuReal combinedContactEffusivity
(
    const GpuReal particleTemperatureK,
    const GpuReal wallDensityKgM3 = defaultWallDensityKgM3,
    const GpuReal wallSpecificHeatJkgK = defaultWallSpecificHeatJkgK,
    const GpuReal wallConductivityWmK = defaultWallConductivityWmK
) noexcept
{
    const GpuReal wall = ::sqrt
    (
        wallConductivityWmK*wallDensityKgM3*wallSpecificHeatJkgK
    );
    if
    (
        !(wallDensityKgM3 > GPU_R(0.0))
     || !(wallSpecificHeatJkgK > GPU_R(0.0))
     || !(wallConductivityWmK > GPU_R(0.0))
     || !finiteContactValue(wall)
    )
    {
        return -GPU_R(1.0);
    }
    return combinedContactEffusivityFromWallEffusivity
    (
        particleTemperatureK, wall
    );
}

                                                                             
                                                                          
inline GpuReal dimensionlessFiniteContactRate
(
    const GpuReal normalizedAge,
    const GpuReal peakTimeFraction,
    const GpuReal normalizedDamage
) noexcept
{
    const GpuReal theta = clampFiniteContact(normalizedAge, GPU_R(0.0), GPU_R(1.0));
    const GpuReal peak = clampFiniteContact(peakTimeFraction, GPU_R(0.0), GPU_R(1.0));
    const GpuReal effectiveArea = normalizedEffectiveArea
    (
        theta, peak, normalizedDamage
    );
    if (!(effectiveArea > GPU_R(0.0)) || !(theta > GPU_R(0.0)) || !(theta < GPU_R(1.0)))
    {
        return GPU_R(0.0);
    }
    const GpuReal radius = ::sqrt(effectiveArea);
    GpuReal integral = GPU_R(0.0);
    if (!(peak > GPU_R(0.0)))
    {
        integral = GPU_R(0.5)*radius*radius/::sqrt(theta);
    }
    else
    {
        const GpuReal lowerZ = ::sqrt
        (
            clampFiniteContact(GPU_R(1.0) - radius*radius, GPU_R(0.0), GPU_R(1.0))
        );
        const GpuReal offset = theta - peak;
        const GpuReal upperQ = theta;
        const GpuReal lowerQ = clampFiniteContact
        (
            offset + peak*lowerZ, GPU_R(0.0), upperQ
        );
        const GpuReal upperRoot = ::sqrt(upperQ);
        const GpuReal lowerRoot = ::sqrt(lowerQ);
        const GpuReal upperPrimitive =
            ((GPU_R(2.0)/GPU_R(3.0))*upperQ - GPU_R(2.0)*offset)*upperRoot;
        const GpuReal lowerPrimitive =
            ((GPU_R(2.0)/GPU_R(3.0))*lowerQ - GPU_R(2.0)*offset)*lowerRoot;
        integral = (upperPrimitive - lowerPrimitive)/(peak*peak);
    }
    return GPU_R(2.0)*finiteContactPi/::sqrt(finiteContactPi)*integral;
}

inline void buildFiniteContactRateTable(double* table) noexcept
{
    if (table == nullptr)
    {
        return;
    }
    for (int peakI = 0; peakI < finiteContactPeakTableSize; ++peakI)
    {
        const GpuReal peak =
            static_cast<GpuReal>(peakI)
           /static_cast<GpuReal>(finiteContactPeakTableSize - 1);
        for (int damageI = 0; damageI < finiteContactDamageTableSize; ++damageI)
        {
            const GpuReal damage =
                static_cast<GpuReal>(damageI)
               /static_cast<GpuReal>(finiteContactDamageTableSize - 1);
            for (int timeI = 0; timeI < finiteContactTimeTableSize; ++timeI)
            {
                const GpuReal theta =
                    static_cast<GpuReal>(timeI)
                   /static_cast<GpuReal>(finiteContactTimeTableSize - 1);
                const int tableIndex =
                    (peakI*finiteContactDamageTableSize + damageI)
                   *finiteContactTimeTableSize + timeI;
                table[tableIndex] = static_cast<double>
                (
                    dimensionlessFiniteContactRate(theta, peak, damage)
                );
            }
        }
    }
}

GPU_FINITE_CONTACT_HD inline GpuReal interpolateFiniteContactRate
(
    const double* table,
    const GpuReal normalizedAge,
    const GpuReal peakTimeFraction,
    const GpuReal normalizedDamage
) noexcept
{
    if (table == nullptr)
    {
        return -GPU_R(1.0);
    }
    const GpuReal timeCoordinate =
        clampFiniteContact(normalizedAge, GPU_R(0.0), GPU_R(1.0))
       *static_cast<GpuReal>(finiteContactTimeTableSize - 1);
    const GpuReal damageCoordinate =
        clampFiniteContact(normalizedDamage, GPU_R(0.0), GPU_R(1.0))
       *static_cast<GpuReal>(finiteContactDamageTableSize - 1);
    const GpuReal peakCoordinate =
        clampFiniteContact(peakTimeFraction, GPU_R(0.0), GPU_R(1.0))
       *static_cast<GpuReal>(finiteContactPeakTableSize - 1);
    const int t0 = static_cast<int>(timeCoordinate);
    const int d0 = static_cast<int>(damageCoordinate);
    const int p0 = static_cast<int>(peakCoordinate);
    const int t1 = t0 + 1 < finiteContactTimeTableSize ? t0 + 1 : t0;
    const int d1 = d0 + 1 < finiteContactDamageTableSize ? d0 + 1 : d0;
    const int p1 = p0 + 1 < finiteContactPeakTableSize ? p0 + 1 : p0;
    const GpuReal ft = timeCoordinate - static_cast<GpuReal>(t0);
    const GpuReal fd = damageCoordinate - static_cast<GpuReal>(d0);
    const GpuReal fp = peakCoordinate - static_cast<GpuReal>(p0);
    const int planeStride =
        finiteContactDamageTableSize*finiteContactTimeTableSize;
    const int p0Offset = p0*planeStride;
    const int p1Offset = p1*planeStride;
    const GpuReal r000 = table[p0Offset + d0*finiteContactTimeTableSize + t0];
    const GpuReal r001 = table[p0Offset + d0*finiteContactTimeTableSize + t1];
    const GpuReal r010 = table[p0Offset + d1*finiteContactTimeTableSize + t0];
    const GpuReal r011 = table[p0Offset + d1*finiteContactTimeTableSize + t1];
    const GpuReal r100 = table[p1Offset + d0*finiteContactTimeTableSize + t0];
    const GpuReal r101 = table[p1Offset + d0*finiteContactTimeTableSize + t1];
    const GpuReal r110 = table[p1Offset + d1*finiteContactTimeTableSize + t0];
    const GpuReal r111 = table[p1Offset + d1*finiteContactTimeTableSize + t1];
    const GpuReal p0Value =
        (GPU_R(1.0) - fd)*((GPU_R(1.0) - ft)*r000 + ft*r001)
      + fd*((GPU_R(1.0) - ft)*r010 + ft*r011);
    const GpuReal p1Value =
        (GPU_R(1.0) - fd)*((GPU_R(1.0) - ft)*r100 + ft*r101)
      + fd*((GPU_R(1.0) - ft)*r110 + ft*r111);
    return (GPU_R(1.0) - fp)*p0Value + fp*p1Value;
}

}                        
}                  

#undef GPU_FINITE_CONTACT_HD

#endif
