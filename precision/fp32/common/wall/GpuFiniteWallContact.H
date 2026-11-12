#include "thermal/ParticleSurfaceState.H"
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
    GPU_R(3.141592653589793238462643383279502884f);
constexpr int finiteContactTimeTableSize = 257;
constexpr int finiteContactDamageTableSize = 65;
constexpr int finiteContactPeakTableSize = 33;
constexpr int finiteContactThermalRingCount = 8;



constexpr GpuReal defaultWallDensityKgM3 = GPU_R(1800.0f);
constexpr GpuReal defaultWallSpecificHeatJkgK = GPU_R(710.0f);
constexpr GpuReal defaultWallConductivityWmK = GPU_R(100.0f);

enum ParticleWallState : unsigned char
{
    particleWallMobile = 0,
    particleWallDeposited = 1,
    particleWallTransientRebound = 2,
    particleWallTransientDeposit = 3
};

// Rebound classification never becomes deposition through freezing alone.
// A frozen contact is released at rest; only a Sommerfeld deposit candidate
// may enter the long-deposit lifecycle.
struct ReboundContactRelease { bool detach; bool atRest; };
GPU_FINITE_CONTACT_HD inline ReboundContactRelease reboundContactRelease
(const GpuReal kinematicArea, const GpuReal frozenArea, const GpuReal damageArea)
{
    return {!(kinematicArea > GPU_R(0.0f))
        || !(std::fmax(kinematicArea,frozenArea)-damageArea > GPU_R(0.0f)),
        frozenArea > GPU_R(0.0f)};
}

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
        GPU_R(2.0f)/(GPU_R(3.0f)*beta)
      + GPU_R(0.25f)*beta*beta*oneMinusCosContactAngle
      - GPU_R(1.0f);
}

GPU_FINITE_CONTACT_HD inline GpuReal correctedSpreadResidual
(
    const GpuReal beta,
    const GpuReal weber,
    const GpuReal ohnesorge,
    const GpuReal oneMinusCosContactAngle
) noexcept
{
    const GpuReal ratio = beta/GPU_R(0.61f);
    const GpuReal ratio2 = ratio*ratio;
    const GpuReal ratio6 = ratio2*ratio2*ratio2;
    return
        ohnesorge*(ratio6 - GPU_R(23.3f))
      - weber
      + GPU_R(12.0f)*finiteContactPhi(beta, oneMinusCosContactAngle);
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
    const GpuReal oneMinusCosContactAngle = GPU_R(1.0f) - contactAngleCosine;

    const GpuReal uncorrected =
        GPU_R(0.61f)*std::pow(GPU_R(23.3f) + weber/ohnesorge, GPU_R(1.0f)/GPU_R(6.0f));
    if (!finiteContactValue(uncorrected))
    {
        return std::numeric_limits<GpuReal>::quiet_NaN();
    }






    if (uncorrected <= GPU_R(1.0f))
    {
        return GPU_R(1.0f);
    }

    GpuReal lower = GPU_R(1.0f);
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
    if (lowerResidual >= GPU_R(0.0f))
    {
        return GPU_R(1.0f);
    }

    int expansion = 0;
    while (!(upperResidual > GPU_R(0.0f)) && expansion < 40)
    {
        lower = upper;
        lowerResidual = upperResidual;
        upper *= GPU_R(2.0f);
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
    if (!(upperResidual > GPU_R(0.0f)))
    {
        return upper;
    }

    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const GpuReal middle = GPU_R(0.5f)*(lower + upper);
        const GpuReal middleResidual = correctedSpreadResidual
        (
            middle, weber, ohnesorge, oneMinusCosContactAngle
        );
        if (!finiteContactValue(middleResidual))
        {
            return std::numeric_limits<GpuReal>::quiet_NaN();
        }
        if (middleResidual <= GPU_R(0.0f))
        {
            lower = middle;
            lowerResidual = middleResidual;
        }
        else
        {
            upper = middle;
        }
    }
    const GpuReal beta = GPU_R(0.5f)*(lower + upper);
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
       /std::sqrt(material.densityKgM3*material.surfaceTensionNM*diameterM);
    const GpuReal beta = correctedMaximumSpreadFactor
    (
        weber, ohnesorge, contactAngleCosine
    );
    const GpuTime spreadingDuration =
        GPU_R(0.5f)*beta*diameterM/normalSpeedMps;
    const GpuTime capillaryTime = std::sqrt
    (
        material.densityKgM3*diameterM*diameterM*diameterM
       /material.surfaceTensionNM
    );
    const GpuReal naturalFrequency = GPU_R(8.0f)/GpuReal(capillaryTime);
    const GpuReal dampingRate =
        GPU_R(20.0f)*material.viscosityPaS
       /(material.densityKgM3*diameterM*diameterM);
    const GpuReal dampedFrequencySquared =
        naturalFrequency*naturalFrequency - dampingRate*dampingRate;

    const GpuReal dampedFrequency = std::sqrt(dampedFrequencySquared);
    const GpuTime recoilDuration =
        (finiteContactPi - std::atan2(dampedFrequency, dampingRate))
       /dampedFrequency;
    const GpuTime duration = spreadingDuration + recoilDuration;
    GpuReal peakTimeFraction = static_cast<GpuReal>(spreadingDuration/duration);
    const GpuReal maximumArea =
        (finiteContactPi/GPU_R(4.0f))*beta*beta*diameterM*diameterM;
    const GpuReal physicalMass =
        (finiteContactPi/GPU_R(6.0f))*material.densityKgM3
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
    if (!(normalizedAge > GPU_R(0.0f)) || normalizedAge >= GPU_R(1.0f))
    {
        return GPU_R(0.0f);
    }
    const GpuReal theta = clampFiniteContact(normalizedAge, GPU_R(0.0f), GPU_R(1.0f));
    const GpuReal peak = clampFiniteContact(peakTimeFraction, GPU_R(0.0f), GPU_R(1.0f));
    if (!(peak > GPU_R(0.0f)))
    {
        const GpuReal cosine = std::cos(GPU_R(0.5f)*finiteContactPi*theta);
        return cosine*cosine;
    }
    if (!(peak < GPU_R(1.0f)))
    {
        return theta*(GPU_R(2.0f) - theta);
    }
    if (theta <= peak)
    {
        const GpuReal x = theta/peak;
        return x*(GPU_R(2.0f) - x);
    }
    const GpuReal y = (theta - peak)/(GPU_R(1.0f) - peak);
    const GpuReal cosine = std::cos(GPU_R(0.5f)*finiteContactPi*y);
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
      - clampFiniteContact(normalizedDamage, GPU_R(0.0f), GPU_R(1.0f));
    return value > GPU_R(0.0f) ? value : GPU_R(0.0f);
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
    if (!(contactAreaM2 > GPU_R(0.0f)))
    {
        return GPU_R(0.0f);
    }
    if (deltaTSeconds == GPU_R(0.0f))
    {
        return GPU_R(0.0f);
    }
    const GpuReal ringArea =
        maximumAreaM2/static_cast<GpuReal>(finiteContactThermalRingCount);
    GpuReal total = GPU_R(0.0f);
    for (int ring = 0; ring < finiteContactThermalRingCount; ++ring)
    {
        const GpuReal ringInnerArea = static_cast<GpuReal>(ring)*ringArea;
        const GpuReal ringWetArea = clampFiniteContact
        (
            contactAreaM2 - ringInnerArea,
            GPU_R(0.0f),
            ringArea
        );
        if (!(ringWetArea > GPU_R(0.0f)))
        {
            continue;
        }
        const GpuReal radialFraction =
            (static_cast<GpuReal>(ring) + GPU_R(0.5f))
           /static_cast<GpuReal>(finiteContactThermalRingCount);
        const GpuReal spreadCoordinate =
            GPU_R(1.0f) - std::sqrt(clampFiniteContact(GPU_R(1.0f) - radialFraction, GPU_R(0.0f), GPU_R(1.0f)));
        const GpuTime firstWetAgeS =
            contactDurationS*peakTimeFraction*spreadCoordinate;
        const GpuTime localAge0S = profileAge0S > firstWetAgeS
          ? profileAge0S - firstWetAgeS
          : GPU_R(0.0f);
        const GpuTime profileAge1S = profileAge0S + deltaTSeconds;
        const GpuTime localAge1S = profileAge1S > firstWetAgeS
          ? profileAge1S - firstWetAgeS
          : GPU_R(0.0f);
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
    if (!material.valid || !(wallEffusivity > GPU_R(0.0f))
     || !finiteContactValue(wallEffusivity))
    {
        return -GPU_R(1.0f);
    }
    const GpuReal particle = std::sqrt
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
    const GpuReal wall = std::sqrt
    (
        wallConductivityWmK*wallDensityKgM3*wallSpecificHeatJkgK
    );
    if
    (
        !(wallDensityKgM3 > GPU_R(0.0f))
     || !(wallSpecificHeatJkgK > GPU_R(0.0f))
     || !(wallConductivityWmK > GPU_R(0.0f))
     || !finiteContactValue(wall)
    )
    {
        return -GPU_R(1.0f);
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
    const GpuReal theta = clampFiniteContact(normalizedAge, GPU_R(0.0f), GPU_R(1.0f));
    const GpuReal peak = clampFiniteContact(peakTimeFraction, GPU_R(0.0f), GPU_R(1.0f));
    const GpuReal effectiveArea = normalizedEffectiveArea
    (
        theta, peak, normalizedDamage
    );
    if (!(effectiveArea > GPU_R(0.0f)) || !(theta > GPU_R(0.0f)) || !(theta < GPU_R(1.0f)))
    {
        return GPU_R(0.0f);
    }
    const GpuReal radius = std::sqrt(effectiveArea);
    GpuReal integral = GPU_R(0.0f);
    if (!(peak > GPU_R(0.0f)))
    {
        integral = GPU_R(0.5f)*radius*radius/std::sqrt(theta);
    }
    else
    {
        const GpuReal lowerZ = std::sqrt
        (
            clampFiniteContact(GPU_R(1.0f) - radius*radius, GPU_R(0.0f), GPU_R(1.0f))
        );
        const GpuReal offset = theta - peak;
        const GpuReal upperQ = theta;
        const GpuReal lowerQ = clampFiniteContact
        (
            offset + peak*lowerZ, GPU_R(0.0f), upperQ
        );
        const GpuReal upperRoot = std::sqrt(upperQ);
        const GpuReal lowerRoot = std::sqrt(lowerQ);
        const GpuReal upperPrimitive =
            ((GPU_R(2.0f)/GPU_R(3.0f))*upperQ - GPU_R(2.0f)*offset)*upperRoot;
        const GpuReal lowerPrimitive =
            ((GPU_R(2.0f)/GPU_R(3.0f))*lowerQ - GPU_R(2.0f)*offset)*lowerRoot;
        integral = (upperPrimitive - lowerPrimitive)/(peak*peak);
    }
    return GPU_R(2.0f)*finiteContactPi/std::sqrt(finiteContactPi)*integral;
}

inline void buildFiniteContactRateTable(float* table) noexcept
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
                table[tableIndex] = static_cast<float>
                (
                    dimensionlessFiniteContactRate(theta, peak, damage)
                );
            }
        }
    }
}

GPU_FINITE_CONTACT_HD inline GpuReal interpolateFiniteContactRate
(
    const float* table,
    const GpuReal normalizedAge,
    const GpuReal peakTimeFraction,
    const GpuReal normalizedDamage
) noexcept
{
    if (table == nullptr)
    {
        return -GPU_R(1.0f);
    }
    const GpuReal timeCoordinate =
        clampFiniteContact(normalizedAge, GPU_R(0.0f), GPU_R(1.0f))
       *static_cast<GpuReal>(finiteContactTimeTableSize - 1);
    const GpuReal damageCoordinate =
        clampFiniteContact(normalizedDamage, GPU_R(0.0f), GPU_R(1.0f))
       *static_cast<GpuReal>(finiteContactDamageTableSize - 1);
    const GpuReal peakCoordinate =
        clampFiniteContact(peakTimeFraction, GPU_R(0.0f), GPU_R(1.0f))
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
        (GPU_R(1.0f) - fd)*((GPU_R(1.0f) - ft)*r000 + ft*r001)
      + fd*((GPU_R(1.0f) - ft)*r010 + ft*r011);
    const GpuReal p1Value =
        (GPU_R(1.0f) - fd)*((GPU_R(1.0f) - ft)*r100 + ft*r101)
      + fd*((GPU_R(1.0f) - ft)*r110 + ft*r111);
    return (GPU_R(1.0f) - fp)*p0Value + fp*p1Value;
}

}
}

#undef GPU_FINITE_CONTACT_HD

#endif
