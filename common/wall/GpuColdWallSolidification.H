#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_COLD_WALL_SOLIDIFICATION_H
#define GPU_THERMAL_GPU_COLD_WALL_SOLIDIFICATION_H

#include "GpuAluminaProperties.H"
#include "GpuWallInterfaceHeat.H"

#include <cmath>
#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#define GPU_COLD_WALL_HD __host__ __device__
#else
#define GPU_COLD_WALL_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr int coldWallAxialNodeCount = 8;
constexpr int coldWallRadialRingCount = 8;
constexpr GpuReal coldWallPi =
    GPU_R(3.141592653589793238462643383279502884);

enum ParticleWallInteractionModel : unsigned char
{
    particleWallInteractionNone = 0,
    particleWallReboundContact = 1,
    particleWallSolidifyingDeposition = 3,
    particleWallColdWall2D = 4
};

struct ColdWallSolidificationParameters
{
    GpuReal meltingTemperatureK;
    GpuReal mushyRangeK;
    GpuReal latentHeatJkg;
    GpuReal solidDensityKgM3;
    GpuReal solidSpecificHeatJkgK;
    GpuReal solidThermalConductivityWmK;
    GpuReal pinningThicknessFraction;
    GpuReal interfaceResistanceM2KW;
    int wallTransientResistance;
    int nonlinearIterations;
};

struct ColdWallSolidificationStep
{
    bool valid;
    GpuReal meanTemperatureK;
    GpuReal surfaceTemperatureK;
    GpuReal wallEnergyJ;
    GpuReal connectedSolidMassKg;
    GpuReal frozenFootprintAreaM2;
};

GPU_COLD_WALL_HD inline bool finiteColdWallValue(const GpuReal value) noexcept
{
    return value == value && value >= -GPU_REAL_MAX && value <= GPU_REAL_MAX;
}

GPU_COLD_WALL_HD inline GpuReal coldWallClamp
(
    const GpuReal value,
    const GpuReal lower,
    const GpuReal upper
) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

GPU_COLD_WALL_HD inline GpuReal coldWallLiquidSpecificHeat
(
    const GpuReal temperatureK
) noexcept
{
    return aluminaSpecificHeatJkgK
    (
        temperatureK < aluminaPropertyTemperatureMinK
      ? aluminaPropertyTemperatureMinK
      : temperatureK
    );
}

GPU_COLD_WALL_HD inline GpuReal coldWallLiquidSensibleIncrement
(
    const GpuReal lowerTemperatureK,
    const GpuReal upperTemperatureK
) noexcept
{
    if (!(upperTemperatureK > lowerTemperatureK))
    {
        return GPU_R(0.0);
    }
    const GpuReal cp0 = coldWallLiquidSpecificHeat(lowerTemperatureK);
    const GpuReal cp1 = coldWallLiquidSpecificHeat(upperTemperatureK);
    return GPU_R(0.5)*(cp0 + cp1)*(upperTemperatureK - lowerTemperatureK);
}

GPU_COLD_WALL_HD inline GpuReal coldWallSolidusTemperature
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return parameters.meltingTemperatureK - GPU_R(0.5)*parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline GpuReal coldWallLiquidusTemperature
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return parameters.meltingTemperatureK + GPU_R(0.5)*parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline GpuReal coldWallMushySpecificHeat
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return
        GPU_R(0.5)*
        (
            parameters.solidSpecificHeatJkgK
          + coldWallLiquidSpecificHeat(parameters.meltingTemperatureK)
        )
      + parameters.latentHeatJkg/parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline GpuReal coldWallSpecificEnthalpyJkg
(
    const GpuReal temperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (!finiteColdWallValue(temperatureK) || temperatureK < GPU_R(0.0))
    {
        return -GPU_R(1.0);
    }
    const GpuReal solidus = coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = coldWallLiquidusTemperature(parameters);
    const GpuReal hSolidus = parameters.solidSpecificHeatJkgK*solidus;
    if (temperatureK <= solidus)
    {
        return parameters.solidSpecificHeatJkgK*temperatureK;
    }
    const GpuReal mushyCp = coldWallMushySpecificHeat(parameters);
    if (temperatureK <= liquidus)
    {
        return hSolidus + mushyCp*(temperatureK - solidus);
    }
    const GpuReal hLiquidus = hSolidus + mushyCp*parameters.mushyRangeK;
    return
        hLiquidus
      + coldWallLiquidSensibleIncrement(liquidus, temperatureK);
}

GPU_COLD_WALL_HD inline GpuReal coldWallTemperatureFromSpecificEnthalpyK
(
    const GpuReal specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (!finiteColdWallValue(specificEnthalpyJkg) || specificEnthalpyJkg < GPU_R(0.0))
    {
        return -GPU_R(1.0);
    }
    const GpuReal solidus = coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = coldWallLiquidusTemperature(parameters);
    const GpuReal hSolidus = parameters.solidSpecificHeatJkgK*solidus;
    if (specificEnthalpyJkg <= hSolidus)
    {
        return specificEnthalpyJkg/parameters.solidSpecificHeatJkgK;
    }
    const GpuReal mushyCp = coldWallMushySpecificHeat(parameters);
    const GpuReal hLiquidus = hSolidus + mushyCp*parameters.mushyRangeK;
    if (specificEnthalpyJkg <= hLiquidus)
    {
        return solidus + (specificEnthalpyJkg - hSolidus)/mushyCp;
    }

    GpuReal lower = liquidus;
    GpuReal upper = liquidus + GPU_R(1000.0);
    while
    (
        coldWallSpecificEnthalpyJkg(upper, parameters)
      < specificEnthalpyJkg
     && upper < GPU_R(20000.0)
    )
    {
        upper *= GPU_R(1.5);
    }
    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const GpuReal middle = GPU_R(0.5)*(lower + upper);
        if
        (
            coldWallSpecificEnthalpyJkg(middle, parameters)
          < specificEnthalpyJkg
        )
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return GPU_R(0.5)*(lower + upper);
}

GPU_COLD_WALL_HD inline GpuReal coldWallSolidFraction
(
    const GpuReal specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const GpuReal temperatureK = coldWallTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg,
        parameters
    );
    const GpuReal solidus = coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus)
    {
        return GPU_R(1.0);
    }
    if (temperatureK >= liquidus)
    {
        return GPU_R(0.0);
    }
    return (liquidus - temperatureK)/parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline GpuReal coldWallApparentSpecificHeat
(
    const GpuReal temperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const GpuReal solidus = coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus)
    {
        return parameters.solidSpecificHeatJkgK;
    }
    if (temperatureK < liquidus)
    {
        return coldWallMushySpecificHeat(parameters);
    }
    return coldWallLiquidSpecificHeat(temperatureK);
}

GPU_COLD_WALL_HD inline GpuReal coldWallThermalConductivity
(
    const GpuReal specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const GpuReal solidFraction = coldWallSolidFraction
    (
        specificEnthalpyJkg,
        parameters
    );
    return
        solidFraction*parameters.solidThermalConductivityWmK
      + (GPU_R(1.0) - solidFraction)*aluminaThermalConductivityWmK;
}

GPU_COLD_WALL_HD inline bool validColdWallSolidificationParameters
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return
        finiteColdWallValue(parameters.meltingTemperatureK)
     && finiteColdWallValue(parameters.mushyRangeK)
     && finiteColdWallValue(parameters.latentHeatJkg)
     && finiteColdWallValue(parameters.solidDensityKgM3)
     && finiteColdWallValue(parameters.solidSpecificHeatJkgK)
     && finiteColdWallValue(parameters.solidThermalConductivityWmK)
     && finiteColdWallValue(parameters.pinningThicknessFraction)
     && finiteColdWallValue(parameters.interfaceResistanceM2KW)
     && parameters.meltingTemperatureK > GPU_R(0.0)
     && parameters.mushyRangeK > GPU_R(0.0)
     && parameters.latentHeatJkg > GPU_R(0.0)
     && parameters.solidDensityKgM3 > GPU_R(0.0)
     && parameters.solidSpecificHeatJkgK > GPU_R(0.0)
     && parameters.solidThermalConductivityWmK > GPU_R(0.0)
     && parameters.pinningThicknessFraction > GPU_R(0.0)
     && parameters.pinningThicknessFraction <= GPU_R(1.0)
     && parameters.interfaceResistanceM2KW >= GPU_R(0.0)
     && (parameters.wallTransientResistance == 0
      || parameters.wallTransientResistance == 1)
     && parameters.nonlinearIterations >= 1
     && parameters.nonlinearIterations <= 12;
}

GPU_COLD_WALL_HD inline void initialiseColdWallProfile
(
    GpuReal* nodeSpecificEnthalpyJkg,
    GpuReal* ringSolidMassKg,
    const GpuReal initialTemperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const GpuReal initialEnthalpy = coldWallSpecificEnthalpyJkg
    (
        initialTemperatureK,
        parameters
    );
    for (int node = 0; node < coldWallAxialNodeCount; ++node)
    {
        nodeSpecificEnthalpyJkg[node] = initialEnthalpy;
    }
    for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
    {
        ringSolidMassKg[ring] = GPU_R(0.0);
    }
}

GPU_COLD_WALL_HD inline void solveColdWallTridiagonal
(
    GpuReal* lower,
    GpuReal* diagonal,
    GpuReal* upper,
    GpuReal* rightHandSide,
    GpuReal* solution
) noexcept
{
    for (int row = 1; row < coldWallAxialNodeCount; ++row)
    {
        const GpuReal multiplier = lower[row]/diagonal[row - 1];
        diagonal[row] -= multiplier*upper[row - 1];
        rightHandSide[row] -= multiplier*rightHandSide[row - 1];
    }
    solution[coldWallAxialNodeCount - 1] =
        rightHandSide[coldWallAxialNodeCount - 1]
       /diagonal[coldWallAxialNodeCount - 1];
    for (int row = coldWallAxialNodeCount - 2; row >= 0; --row)
    {
        solution[row] =
            (rightHandSide[row] - upper[row]*solution[row + 1])
           /diagonal[row];
    }
}

GPU_COLD_WALL_HD inline ColdWallSolidificationStep advanceColdWallProfile
(
    GpuReal* nodeSpecificEnthalpyJkg,
    GpuReal* ringSolidMassKg,
    GpuTime& profileContactAgeS,
    GpuReal& frozenFootprintAreaM2,
    const ColdWallSolidificationParameters& parameters,
    const GpuReal particleVolumeM3,
    const GpuReal parcelMassKg,
    const GpuReal maximumAreaM2,
    const GpuReal intrinsicContactAreaM2,
    const GpuTime contactDurationS,
    const GpuReal peakTimeFraction,
    const GpuTime deltaTSeconds,
    const GpuReal wallTemperatureK,
    const GpuReal wallEffusivity,
    const GpuReal thermalAreaFactor,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK
) noexcept
{
    if
    (
        nodeSpecificEnthalpyJkg == nullptr
     || ringSolidMassKg == nullptr
     || !validColdWallSolidificationParameters(parameters)
     || !(particleVolumeM3 > GPU_R(0.0))
     || !(parcelMassKg > GPU_R(0.0))
     || !(maximumAreaM2 > GPU_R(0.0))
     || !(intrinsicContactAreaM2 > GPU_R(0.0))
     || !(contactDurationS > GPU_R(0.0))
     || !(peakTimeFraction > GPU_R(0.0))
     || !(peakTimeFraction < GPU_R(1.0))
     || !(deltaTSeconds >= GPU_R(0.0))
     || !(wallTemperatureK > GPU_R(0.0))
     || !(wallEffusivity > GPU_R(0.0))
     || !(thermalAreaFactor > GPU_R(0.0))
     || thermalAreaFactor > GPU_R(1.0)
     || !(gasTemperatureK > GPU_R(0.0))
     || gasConductanceWK < GPU_R(0.0)
    )
    {
        return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    }

    const GpuTime ageNew = profileContactAgeS + deltaTSeconds;
    const GpuReal contactArea = coldWallClamp
    (
        intrinsicContactAreaM2,
        GPU_REAL_MIN,
        maximumAreaM2
    );
    const GpuReal filmThickness = particleVolumeM3/contactArea;
    const GpuReal nodeThickness =
        filmThickness/static_cast<GpuReal>(coldWallAxialNodeCount);
    const GpuReal nodeMass =
        parcelMassKg/static_cast<GpuReal>(coldWallAxialNodeCount);
    const GpuReal ringArea =
        maximumAreaM2/static_cast<GpuReal>(coldWallRadialRingCount);
    GpuReal oldEnthalpy[coldWallAxialNodeCount];
    GpuReal guessedTemperature[coldWallAxialNodeCount];
    GpuReal candidateEnthalpy[coldWallAxialNodeCount];
    GpuReal faceConductance[coldWallAxialNodeCount - 1];
    GpuReal ringCoolingPower[coldWallRadialRingCount];
    GpuReal ringWetArea[coldWallRadialRingCount];
    for (int node = 0; node < coldWallAxialNodeCount; ++node)
    {
        oldEnthalpy[node] = nodeSpecificEnthalpyJkg[node];
        guessedTemperature[node] = coldWallTemperatureFromSpecificEnthalpyK
        (
            oldEnthalpy[node],
            parameters
        );
        candidateEnthalpy[node] = oldEnthalpy[node];
    }

    GpuReal wallConductance = GPU_R(0.0);
    GpuReal acceptedWallPower = GPU_R(0.0);
    for (int iteration = 0; iteration < parameters.nonlinearIterations; ++iteration)
    {
        wallConductance = GPU_R(0.0);
        const GpuReal bottomConductivity = coldWallThermalConductivity
        (
            candidateEnthalpy[0],
            parameters
        );
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const GpuReal ringInnerArea = static_cast<GpuReal>(ring)*ringArea;
            const GpuReal wetArea = coldWallClamp
            (
                contactArea - ringInnerArea,
                GPU_R(0.0),
                ringArea
            );
            ringWetArea[ring] = wetArea;
            if (!(wetArea > GPU_R(0.0)))
            {
                ringCoolingPower[ring] = GPU_R(0.0);
                continue;
            }
            const GpuReal radialFraction =
                (static_cast<GpuReal>(ring) + GPU_R(0.5))
               /static_cast<GpuReal>(coldWallRadialRingCount);
            const GpuReal spreadCoordinate =
                GPU_R(1.0) - ::sqrt(coldWallClamp(GPU_R(1.0) - radialFraction, GPU_R(0.0), GPU_R(1.0)));
            const GpuTime firstWetAge =
                contactDurationS*peakTimeFraction*spreadCoordinate;
            const GpuTime localAge0 =
                profileContactAgeS > firstWetAge
              ? profileContactAgeS - firstWetAge
              : GPU_R(0.0);
            const GpuTime localAge1 =
                ageNew > firstWetAge ? ageNew - firstWetAge : GPU_R(0.0);
            const GpuReal particleResistance =
                GPU_R(0.5)*nodeThickness/bottomConductivity;
            const GpuReal conductanceIntegral =
                wallInterfaceConductanceTimeIntegral
                (
                    wetArea,
                    thermalAreaFactor,
                    localAge0,
                    localAge1,
                    parameters.interfaceResistanceM2KW,
                    particleResistance,
                    wallEffusivity,
                    parameters.wallTransientResistance != 0
                );
            const GpuReal conductance =
                conductanceIntegral/(deltaTSeconds + GPU_REAL_MIN);
            if (!(conductanceIntegral >= GPU_R(0.0)))
            {
                return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
            }
            wallConductance += conductance;
            ringCoolingPower[ring] = conductance*
                (guessedTemperature[0] - wallTemperatureK);
        }

        for (int face = 0; face < coldWallAxialNodeCount - 1; ++face)
        {
            const GpuReal leftK = coldWallThermalConductivity
            (
                candidateEnthalpy[face],
                parameters
            );
            const GpuReal rightK = coldWallThermalConductivity
            (
                candidateEnthalpy[face + 1],
                parameters
            );
            const GpuReal harmonicK = GPU_R(2.0)*leftK*rightK/(leftK + rightK);
            faceConductance[face] =
                harmonicK*contactArea/nodeThickness;
        }

        GpuReal lower[coldWallAxialNodeCount];
        GpuReal diagonal[coldWallAxialNodeCount];
        GpuReal upper[coldWallAxialNodeCount];
        GpuReal rightHandSide[coldWallAxialNodeCount];
        GpuReal solvedTemperature[coldWallAxialNodeCount];
        for (int node = 0; node < coldWallAxialNodeCount; ++node)
        {
            lower[node] = GPU_R(0.0);
            upper[node] = GPU_R(0.0);
            const GpuReal capacity =
                nodeMass*coldWallApparentSpecificHeat
                (
                    guessedTemperature[node],
                    parameters
                )/(deltaTSeconds + GPU_REAL_MIN);
            diagonal[node] = capacity;
            rightHandSide[node] = capacity*guessedTemperature[node];
            if (node > 0)
            {
                lower[node] = -faceConductance[node - 1];
                diagonal[node] += faceConductance[node - 1];
            }
            if (node + 1 < coldWallAxialNodeCount)
            {
                upper[node] = -faceConductance[node];
                diagonal[node] += faceConductance[node];
            }
        }
        diagonal[0] += wallConductance;
        rightHandSide[0] += wallConductance*wallTemperatureK;
        diagonal[coldWallAxialNodeCount - 1] += gasConductanceWK;
        rightHandSide[coldWallAxialNodeCount - 1] +=
            gasConductanceWK*gasTemperatureK;
        solveColdWallTridiagonal
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            solvedTemperature
        );

        GpuReal internalPower[coldWallAxialNodeCount - 1];
        for (int face = 0; face < coldWallAxialNodeCount - 1; ++face)
        {
            internalPower[face] = faceConductance[face]
               *(solvedTemperature[face] - solvedTemperature[face + 1]);
        }
        const GpuReal wallPower = wallConductance
           *(solvedTemperature[0] - wallTemperatureK);
        acceptedWallPower = wallPower;
        const GpuReal gasPower = gasConductanceWK
           *(gasTemperatureK - solvedTemperature[coldWallAxialNodeCount - 1]);
        for (int node = 0; node < coldWallAxialNodeCount; ++node)
        {
            GpuReal power = GPU_R(0.0);
            if (node == 0)
            {
                power -= wallPower;
            }
            if (node > 0)
            {
                power += internalPower[node - 1];
            }
            if (node + 1 < coldWallAxialNodeCount)
            {
                power -= internalPower[node];
            }
            if (node == coldWallAxialNodeCount - 1)
            {
                power += gasPower;
            }
            candidateEnthalpy[node] = oldEnthalpy[node]
              + deltaTSeconds*power/nodeMass;
            if
            (
                !finiteColdWallValue(candidateEnthalpy[node])
             || candidateEnthalpy[node] < GPU_R(0.0)
            )
            {
                return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
            }
            guessedTemperature[node] = coldWallTemperatureFromSpecificEnthalpyK
            (
                candidateEnthalpy[node],
                parameters
            );
        }
    }

    GpuReal enthalpySum = GPU_R(0.0);
    GpuReal connectedFraction = GPU_R(1.0);
    GpuReal connectedMass = GPU_R(0.0);
    for (int node = 0; node < coldWallAxialNodeCount; ++node)
    {
        nodeSpecificEnthalpyJkg[node] = candidateEnthalpy[node];
        enthalpySum += candidateEnthalpy[node];
        const GpuReal solidFraction = coldWallSolidFraction
        (
            candidateEnthalpy[node],
            parameters
        );
        connectedFraction =
            solidFraction < connectedFraction ? solidFraction : connectedFraction;
        connectedMass += nodeMass*connectedFraction;
    }

    GpuReal assignedMass = GPU_R(0.0);
    for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
    {
        assignedMass += ringSolidMassKg[ring];
    }
    if (connectedMass < assignedMass && assignedMass > GPU_R(0.0))
    {
        const GpuReal scale = connectedMass/assignedMass;
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            ringSolidMassKg[ring] *= scale;
        }
        assignedMass = connectedMass;
    }
    GpuReal remainingMass = connectedMass - assignedMass;
    for
    (
        int pass = 0;
        pass < coldWallRadialRingCount && remainingMass > GPU_R(1.0e-18)*parcelMassKg;
        ++pass
    )
    {
        GpuReal weightSum = GPU_R(0.0);
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const GpuReal capacity =
                parameters.solidDensityKgM3*ringWetArea[ring]*filmThickness;
            if (capacity > ringSolidMassKg[ring])
            {
                const GpuReal cooling = ringCoolingPower[ring] > GPU_R(0.0)
                  ? ringCoolingPower[ring] : GPU_R(0.0);
                weightSum += cooling > GPU_R(0.0) ? cooling : ringWetArea[ring];
            }
        }
        if (!(weightSum > GPU_R(0.0)))
        {
            break;
        }
        GpuReal allocatedThisPass = GPU_R(0.0);
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const GpuReal capacity =
                parameters.solidDensityKgM3*ringWetArea[ring]*filmThickness;
            const GpuReal available = capacity - ringSolidMassKg[ring];
            if (!(available > GPU_R(0.0)))
            {
                continue;
            }
            const GpuReal cooling = ringCoolingPower[ring] > GPU_R(0.0)
              ? ringCoolingPower[ring] : GPU_R(0.0);
            const GpuReal weight = cooling > GPU_R(0.0) ? cooling : ringWetArea[ring];
            const GpuReal requested = remainingMass*weight/weightSum;
            const GpuReal addition = requested < available ? requested : available;
            ringSolidMassKg[ring] += addition;
            allocatedThisPass += addition;
        }
        remainingMass -= allocatedThisPass;
        if (!(allocatedThisPass > GPU_R(0.0)))
        {
            break;
        }
    }

    GpuReal candidateFrozenArea = GPU_R(0.0);
    for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
    {
        if (!(ringWetArea[ring] > GPU_R(0.0)))
        {
            break;
        }
        const GpuReal localSolidFraction = ringSolidMassKg[ring]
           /(parameters.solidDensityKgM3*ringWetArea[ring]*filmThickness);
        if (localSolidFraction < parameters.pinningThicknessFraction)
        {
            break;
        }
        candidateFrozenArea += ringWetArea[ring];
    }
    if (candidateFrozenArea > frozenFootprintAreaM2)
    {
        frozenFootprintAreaM2 = candidateFrozenArea;
    }
    profileContactAgeS = ageNew;

    const GpuReal meanEnthalpy =
        enthalpySum/static_cast<GpuReal>(coldWallAxialNodeCount);
    const GpuReal meanTemperature = coldWallTemperatureFromSpecificEnthalpyK
    (
        meanEnthalpy,
        parameters
    );
    const GpuReal surfaceTemperature = coldWallTemperatureFromSpecificEnthalpyK
    (
        candidateEnthalpy[coldWallAxialNodeCount - 1],
        parameters
    );
    const GpuReal wallEnergy = acceptedWallPower*deltaTSeconds;
    const bool valid =
        finiteColdWallValue(meanTemperature)
     && finiteColdWallValue(surfaceTemperature)
     && finiteColdWallValue(wallEnergy)
     && finiteColdWallValue(connectedMass)
     && finiteColdWallValue(frozenFootprintAreaM2)
     && meanTemperature > GPU_R(0.0)
     && surfaceTemperature > GPU_R(0.0)
     && connectedMass >= GPU_R(0.0)
     && frozenFootprintAreaM2 >= GPU_R(0.0);
    return
    {
        valid,
        valid ? meanTemperature : GPU_R(0.0),
        valid ? surfaceTemperature : GPU_R(0.0),
        valid ? wallEnergy : GPU_R(0.0),
        valid ? connectedMass : GPU_R(0.0),
        valid ? frozenFootprintAreaM2 : GPU_R(0.0)
    };
}

}
}

#undef GPU_COLD_WALL_HD

#endif
