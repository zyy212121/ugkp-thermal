#ifndef GPU_THERMAL_GPU_FINITE_WALL_CONTACT_H
#define GPU_THERMAL_GPU_FINITE_WALL_CONTACT_H

#include "GpuAluminaProperties.H"
#include "GpuWallInterfaceHeat.H"

#include <cmath>
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

constexpr double finiteContactPi =
    3.141592653589793238462643383279502884;
constexpr int finiteContactTimeTableSize = 257;
constexpr int finiteContactDamageTableSize = 65;
constexpr int finiteContactPeakTableSize = 33;
constexpr int finiteContactThermalRingCount = 8;

                                                                           
                                                                              
constexpr double defaultWallDensityKgM3 = 1800.0;
constexpr double defaultWallSpecificHeatJkgK = 710.0;
constexpr double defaultWallConductivityWmK = 100.0;

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
    double betaMax;
    double maximumAreaM2;
    double contactDurationS;
    double peakTimeFraction;
    double physicalMassKg;
    double weber;
    double ohnesorge;
};

GPU_FINITE_CONTACT_HD inline bool finiteContactValue(const double x) noexcept
{
    return x == x && x >= -DBL_MAX && x <= DBL_MAX;
}

GPU_FINITE_CONTACT_HD inline double clampFiniteContact
(
    const double x,
    const double lo,
    const double hi
) noexcept
{
    return x < lo ? lo : (x > hi ? hi : x);
}

GPU_FINITE_CONTACT_HD inline double finiteContactPhi
(
    const double beta,
    const double oneMinusCosContactAngle
) noexcept
{
    return
        2.0/(3.0*beta)
      + 0.25*beta*beta*oneMinusCosContactAngle
      - 1.0;
}

GPU_FINITE_CONTACT_HD inline double correctedSpreadResidual
(
    const double beta,
    const double weber,
    const double ohnesorge,
    const double oneMinusCosContactAngle
) noexcept
{
    const double ratio = beta/0.61;
    const double ratio2 = ratio*ratio;
    const double ratio6 = ratio2*ratio2*ratio2;
    return
        ohnesorge*(ratio6 - 23.3)
      - weber
      + 12.0*finiteContactPhi(beta, oneMinusCosContactAngle);
}

GPU_FINITE_CONTACT_HD inline double correctedMaximumSpreadFactor
(
    const double weber,
    const double ohnesorge,
    const double contactAngleCosine
) noexcept
{
    if
    (
        !(weber >= 0.0)
     || !(ohnesorge > 0.0)
     || !finiteContactValue(contactAngleCosine)
     || !(contactAngleCosine > -1.0)
     || !(contactAngleCosine < 1.0)
    )
    {
        return -1.0;
    }
    const double oneMinusCosContactAngle = 1.0 - contactAngleCosine;

    const double uncorrected =
        0.61*::pow(23.3 + weber/ohnesorge, 1.0/6.0);
    if (!(uncorrected > 0.0) || !finiteContactValue(uncorrected))
    {
        return -1.0;
    }

                                                                         
                                                                     
                                                                           
                                                                         
                                                                            
    if (uncorrected <= 1.0)
    {
        return 1.0;
    }

    double lower = 1.0;
    double upper = uncorrected;
    double lowerResidual = correctedSpreadResidual
    (
        lower, weber, ohnesorge, oneMinusCosContactAngle
    );
    double upperResidual = correctedSpreadResidual
    (
        upper, weber, ohnesorge, oneMinusCosContactAngle
    );
    if
    (
        !finiteContactValue(lowerResidual)
     || !finiteContactValue(upperResidual)
    )
    {
        return -1.0;
    }
    if (lowerResidual >= 0.0)
    {
        return 1.0;
    }

    int expansion = 0;
    while (!(upperResidual > 0.0) && expansion < 40)
    {
        lower = upper;
        lowerResidual = upperResidual;
        upper *= 2.0;
        if (!finiteContactValue(upper))
        {
            return -1.0;
        }
        upperResidual = correctedSpreadResidual
        (
            upper, weber, ohnesorge, oneMinusCosContactAngle
        );
        if (!finiteContactValue(upperResidual))
        {
            return -1.0;
        }
        ++expansion;
    }
    if (!(upperResidual > 0.0))
    {
        return -1.0;
    }

    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const double middle = 0.5*(lower + upper);
        const double middleResidual = correctedSpreadResidual
        (
            middle, weber, ohnesorge, oneMinusCosContactAngle
        );
        if (!finiteContactValue(middleResidual))
        {
            return -1.0;
        }
        if (middleResidual <= 0.0)
        {
            lower = middle;
            lowerResidual = middleResidual;
        }
        else
        {
            upper = middle;
        }
    }
    const double beta = 0.5*(lower + upper);
    return finiteContactValue(beta) && beta >= 1.0 ? beta : -1.0;
}

GPU_FINITE_CONTACT_HD inline FiniteWallContactImpact evaluateFiniteWallContactImpact
(
    const double particleTemperatureK,
    const double diameterM,
    const double normalSpeedMps,
    const double contactAngleCosine
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if
    (
        !material.valid || !(diameterM > 0.0) || !(normalSpeedMps > 0.0)
    )
    {
        return {false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double weber =
        material.densityKgM3*normalSpeedMps*normalSpeedMps*diameterM
       /material.surfaceTensionNM;
    const double ohnesorge =
        material.viscosityPaS
       /::sqrt(material.densityKgM3*material.surfaceTensionNM*diameterM);
    const double beta = correctedMaximumSpreadFactor
    (
        weber, ohnesorge, contactAngleCosine
    );
    const double spreadingDuration =
        0.5*beta*diameterM/normalSpeedMps;
    const double capillaryTime = ::sqrt
    (
        material.densityKgM3*diameterM*diameterM*diameterM
       /material.surfaceTensionNM
    );
    const double naturalFrequency = 8.0/capillaryTime;
    const double dampingRate =
        20.0*material.viscosityPaS
       /(material.densityKgM3*diameterM*diameterM);
    const double dampedFrequencySquared =
        naturalFrequency*naturalFrequency - dampingRate*dampingRate;
    if (!(dampedFrequencySquared > 0.0))
    {
        return {false, 0.0, 0.0, 0.0, 0.0, 0.0, weber, ohnesorge};
    }
    const double dampedFrequency = ::sqrt(dampedFrequencySquared);
    const double recoilDuration =
        (finiteContactPi - ::atan2(dampedFrequency, dampingRate))
       /dampedFrequency;
    const double duration = spreadingDuration + recoilDuration;
    const double peakTimeFraction = spreadingDuration/duration;
    const double maximumArea =
        (finiteContactPi/4.0)*beta*beta*diameterM*diameterM;
    const double physicalMass =
        (finiteContactPi/6.0)*material.densityKgM3
       *diameterM*diameterM*diameterM;
    const bool valid =
        beta > 0.0 && spreadingDuration > 0.0 && recoilDuration > 0.0
     && duration > 0.0 && peakTimeFraction > 0.0 && peakTimeFraction < 1.0
     && maximumArea > 0.0
     && physicalMass > 0.0 && finiteContactValue(duration)
     && finiteContactValue(peakTimeFraction)
     && finiteContactValue(maximumArea);
    return
    {
        valid,
        valid ? beta : 0.0,
        valid ? maximumArea : 0.0,
        valid ? duration : 0.0,
        valid ? peakTimeFraction : 0.0,
        valid ? physicalMass : 0.0,
        weber,
        ohnesorge
    };
}

GPU_FINITE_CONTACT_HD inline double normalizedKinematicArea
(
    const double normalizedAge,
    const double peakTimeFraction
) noexcept
{
    if (!(normalizedAge > 0.0) || normalizedAge >= 1.0)
    {
        return 0.0;
    }
    const double theta = clampFiniteContact(normalizedAge, 0.0, 1.0);
    const double peak = clampFiniteContact(peakTimeFraction, 0.0, 1.0);
    if (!(peak > 0.0))
    {
        const double cosine = ::cos(0.5*finiteContactPi*theta);
        return cosine*cosine;
    }
    if (!(peak < 1.0))
    {
        return theta*(2.0 - theta);
    }
    if (theta <= peak)
    {
        const double x = theta/peak;
        return x*(2.0 - x);
    }
    const double y = (theta - peak)/(1.0 - peak);
    const double cosine = ::cos(0.5*finiteContactPi*y);
    return cosine*cosine;
}

GPU_FINITE_CONTACT_HD inline double normalizedEffectiveArea
(
    const double normalizedAge,
    const double peakTimeFraction,
    const double normalizedDamage
) noexcept
{
    const double value =
        normalizedKinematicArea(normalizedAge, peakTimeFraction)
      - clampFiniteContact(normalizedDamage, 0.0, 1.0);
    return value > 0.0 ? value : 0.0;
}

GPU_FINITE_CONTACT_HD inline double finiteContactWallConductanceTimeIntegral
(
    const double maximumAreaM2,
    const double contactAreaM2,
    const double contactDurationS,
    const double peakTimeFraction,
    const double profileAge0S,
    const double deltaTSeconds,
    const double efficiency,
    const double interfaceResistanceM2KW,
    const double particleAdditionalResistanceM2KW,
    const double wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if
    (
        !finiteContactValue(maximumAreaM2)
     || !finiteContactValue(contactAreaM2)
     || !finiteContactValue(contactDurationS)
     || !finiteContactValue(peakTimeFraction)
     || !finiteContactValue(profileAge0S)
     || !finiteContactValue(deltaTSeconds)
     || !(maximumAreaM2 > 0.0)
     || contactAreaM2 < 0.0
     || contactAreaM2 > maximumAreaM2
     || !(contactDurationS > 0.0)
     || !(peakTimeFraction > 0.0)
     || !(peakTimeFraction < 1.0)
     || profileAge0S < 0.0
     || deltaTSeconds < 0.0
    )
    {
        return -1.0;
    }
    if (!(contactAreaM2 > 0.0))
    {
        return 0.0;
    }
    if (!(deltaTSeconds > 0.0))
    {
        return 0.0;
    }
    const double ringArea =
        maximumAreaM2/static_cast<double>(finiteContactThermalRingCount);
    double total = 0.0;
    for (int ring = 0; ring < finiteContactThermalRingCount; ++ring)
    {
        const double ringInnerArea = static_cast<double>(ring)*ringArea;
        const double ringWetArea = clampFiniteContact
        (
            contactAreaM2 - ringInnerArea,
            0.0,
            ringArea
        );
        if (!(ringWetArea > 0.0))
        {
            continue;
        }
        const double radialFraction =
            (static_cast<double>(ring) + 0.5)
           /static_cast<double>(finiteContactThermalRingCount);
        const double spreadCoordinate =
            1.0 - ::sqrt(clampFiniteContact(1.0 - radialFraction, 0.0, 1.0));
        const double firstWetAgeS =
            contactDurationS*peakTimeFraction*spreadCoordinate;
        const double localAge0S = profileAge0S > firstWetAgeS
          ? profileAge0S - firstWetAgeS
          : 0.0;
        const double profileAge1S = profileAge0S + deltaTSeconds;
        const double localAge1S = profileAge1S > firstWetAgeS
          ? profileAge1S - firstWetAgeS
          : 0.0;
        const double contribution = wallInterfaceConductanceTimeIntegral
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
        if (!(contribution >= 0.0) || !finiteContactValue(contribution))
        {
            return -1.0;
        }
        total += contribution;
    }
    return finiteContactValue(total) && total >= 0.0 ? total : -1.0;
}

GPU_FINITE_CONTACT_HD inline double combinedContactEffusivityFromWallEffusivity
(
    const double particleTemperatureK,
    const double wallEffusivity
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if (!material.valid || !(wallEffusivity > 0.0)
     || !finiteContactValue(wallEffusivity))
    {
        return -1.0;
    }
    const double particle = ::sqrt
    (
        material.thermalConductivityWmK*material.densityKgM3
       *material.specificHeatJkgK
    );
    return particle*wallEffusivity/(particle + wallEffusivity);
}

GPU_FINITE_CONTACT_HD inline double combinedContactEffusivity
(
    const double particleTemperatureK,
    const double wallDensityKgM3 = defaultWallDensityKgM3,
    const double wallSpecificHeatJkgK = defaultWallSpecificHeatJkgK,
    const double wallConductivityWmK = defaultWallConductivityWmK
) noexcept
{
    const double wall = ::sqrt
    (
        wallConductivityWmK*wallDensityKgM3*wallSpecificHeatJkgK
    );
    if
    (
        !(wallDensityKgM3 > 0.0)
     || !(wallSpecificHeatJkgK > 0.0)
     || !(wallConductivityWmK > 0.0)
     || !finiteContactValue(wall)
    )
    {
        return -1.0;
    }
    return combinedContactEffusivityFromWallEffusivity
    (
        particleTemperatureK, wall
    );
}

                                                                             
                                                                          
inline double dimensionlessFiniteContactRate
(
    const double normalizedAge,
    const double peakTimeFraction,
    const double normalizedDamage
) noexcept
{
    const double theta = clampFiniteContact(normalizedAge, 0.0, 1.0);
    const double peak = clampFiniteContact(peakTimeFraction, 0.0, 1.0);
    const double effectiveArea = normalizedEffectiveArea
    (
        theta, peak, normalizedDamage
    );
    if (!(effectiveArea > 0.0) || !(theta > 0.0) || !(theta < 1.0))
    {
        return 0.0;
    }
    const double radius = ::sqrt(effectiveArea);
    double integral = 0.0;
    if (!(peak > 0.0))
    {
        integral = 0.5*radius*radius/::sqrt(theta);
    }
    else
    {
        const double lowerZ = ::sqrt
        (
            clampFiniteContact(1.0 - radius*radius, 0.0, 1.0)
        );
        const double offset = theta - peak;
        const double upperQ = theta;
        const double lowerQ = clampFiniteContact
        (
            offset + peak*lowerZ, 0.0, upperQ
        );
        const double upperRoot = ::sqrt(upperQ);
        const double lowerRoot = ::sqrt(lowerQ);
        const double upperPrimitive =
            ((2.0/3.0)*upperQ - 2.0*offset)*upperRoot;
        const double lowerPrimitive =
            ((2.0/3.0)*lowerQ - 2.0*offset)*lowerRoot;
        integral = (upperPrimitive - lowerPrimitive)/(peak*peak);
    }
    return 2.0*finiteContactPi/::sqrt(finiteContactPi)*integral;
}

inline void buildFiniteContactRateTable(float* table) noexcept
{
    if (table == nullptr)
    {
        return;
    }
    for (int peakI = 0; peakI < finiteContactPeakTableSize; ++peakI)
    {
        const double peak =
            static_cast<double>(peakI)
           /static_cast<double>(finiteContactPeakTableSize - 1);
        for (int damageI = 0; damageI < finiteContactDamageTableSize; ++damageI)
        {
            const double damage =
                static_cast<double>(damageI)
               /static_cast<double>(finiteContactDamageTableSize - 1);
            for (int timeI = 0; timeI < finiteContactTimeTableSize; ++timeI)
            {
                const double theta =
                    static_cast<double>(timeI)
                   /static_cast<double>(finiteContactTimeTableSize - 1);
                const int tableIndex =
                    (peakI*finiteContactDamageTableSize + damageI)
                   *finiteContactTimeTableSize + timeI;
                table[tableIndex] = static_cast<float>
                (
                    dimensionlessFiniteContactRate(theta, peak, damage)
                );
            }
        }
    }
}

GPU_FINITE_CONTACT_HD inline double interpolateFiniteContactRate
(
    const float* table,
    const double normalizedAge,
    const double peakTimeFraction,
    const double normalizedDamage
) noexcept
{
    if (table == nullptr)
    {
        return -1.0;
    }
    const double timeCoordinate =
        clampFiniteContact(normalizedAge, 0.0, 1.0)
       *static_cast<double>(finiteContactTimeTableSize - 1);
    const double damageCoordinate =
        clampFiniteContact(normalizedDamage, 0.0, 1.0)
       *static_cast<double>(finiteContactDamageTableSize - 1);
    const double peakCoordinate =
        clampFiniteContact(peakTimeFraction, 0.0, 1.0)
       *static_cast<double>(finiteContactPeakTableSize - 1);
    const int t0 = static_cast<int>(timeCoordinate);
    const int d0 = static_cast<int>(damageCoordinate);
    const int p0 = static_cast<int>(peakCoordinate);
    const int t1 = t0 + 1 < finiteContactTimeTableSize ? t0 + 1 : t0;
    const int d1 = d0 + 1 < finiteContactDamageTableSize ? d0 + 1 : d0;
    const int p1 = p0 + 1 < finiteContactPeakTableSize ? p0 + 1 : p0;
    const double ft = timeCoordinate - static_cast<double>(t0);
    const double fd = damageCoordinate - static_cast<double>(d0);
    const double fp = peakCoordinate - static_cast<double>(p0);
    const int planeStride =
        finiteContactDamageTableSize*finiteContactTimeTableSize;
    const int p0Offset = p0*planeStride;
    const int p1Offset = p1*planeStride;
    const double r000 = table[p0Offset + d0*finiteContactTimeTableSize + t0];
    const double r001 = table[p0Offset + d0*finiteContactTimeTableSize + t1];
    const double r010 = table[p0Offset + d1*finiteContactTimeTableSize + t0];
    const double r011 = table[p0Offset + d1*finiteContactTimeTableSize + t1];
    const double r100 = table[p1Offset + d0*finiteContactTimeTableSize + t0];
    const double r101 = table[p1Offset + d0*finiteContactTimeTableSize + t1];
    const double r110 = table[p1Offset + d1*finiteContactTimeTableSize + t0];
    const double r111 = table[p1Offset + d1*finiteContactTimeTableSize + t1];
    const double p0Value =
        (1.0 - fd)*((1.0 - ft)*r000 + ft*r001)
      + fd*((1.0 - ft)*r010 + ft*r011);
    const double p1Value =
        (1.0 - fd)*((1.0 - ft)*r100 + ft*r101)
      + fd*((1.0 - ft)*r110 + ft*r111);
    return (1.0 - fp)*p0Value + fp*p1Value;
}

}                        
}                  

#undef GPU_FINITE_CONTACT_HD

#endif
