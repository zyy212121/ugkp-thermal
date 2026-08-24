#ifndef GPU_THERMAL_GPU_COLD_WALL_SOLIDIFICATION_H
#define GPU_THERMAL_GPU_COLD_WALL_SOLIDIFICATION_H

#include "GpuAluminaProperties.H"

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
constexpr double coldWallPi =
    3.141592653589793238462643383279502884;

enum ParticleWallInteractionModel : unsigned char
{
    particleWallInteractionNone = 0,
    particleWallReboundContact = 1,
    particleWallLiquidDeposition = 2,
    particleWallSolidifyingDeposition = 3,
    particleWallColdWall2D = 4
};

struct ColdWallSolidificationParameters
{
    double meltingTemperatureK;
    double mushyRangeK;
    double latentHeatJkg;
    double solidDensityKgM3;
    double solidSpecificHeatJkgK;
    double solidThermalConductivityWmK;
    double pinningThicknessFraction;
    double interfaceResistanceM2KW;
    int wallTransientResistance;
    int nonlinearIterations;
};

struct ColdWallSolidificationStep
{
    bool valid;
    double meanTemperatureK;
    double surfaceTemperatureK;
    double wallEnergyJ;
    double connectedSolidMassKg;
    double frozenFootprintAreaM2;
};

GPU_COLD_WALL_HD inline bool finiteColdWallValue(const double value) noexcept
{
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

GPU_COLD_WALL_HD inline double coldWallClamp
(
    const double value,
    const double lower,
    const double upper
) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

GPU_COLD_WALL_HD inline double coldWallLiquidSpecificHeat
(
    const double temperatureK
) noexcept
{
    return aluminaSpecificHeatJkgK
    (
        temperatureK < aluminaPropertyTemperatureMinK
      ? aluminaPropertyTemperatureMinK
      : temperatureK
    );
}

GPU_COLD_WALL_HD inline double coldWallLiquidSensibleIncrement
(
    const double lowerTemperatureK,
    const double upperTemperatureK
) noexcept
{
    if (!(upperTemperatureK > lowerTemperatureK))
    {
        return 0.0;
    }
    const double cp0 = coldWallLiquidSpecificHeat(lowerTemperatureK);
    const double cp1 = coldWallLiquidSpecificHeat(upperTemperatureK);
    return 0.5*(cp0 + cp1)*(upperTemperatureK - lowerTemperatureK);
}

GPU_COLD_WALL_HD inline double coldWallSolidusTemperature
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return parameters.meltingTemperatureK - 0.5*parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline double coldWallLiquidusTemperature
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return parameters.meltingTemperatureK + 0.5*parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline double coldWallMushySpecificHeat
(
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    return
        0.5*
        (
            parameters.solidSpecificHeatJkgK
          + coldWallLiquidSpecificHeat(parameters.meltingTemperatureK)
        )
      + parameters.latentHeatJkg/parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline double coldWallSpecificEnthalpyJkg
(
    const double temperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (!finiteColdWallValue(temperatureK) || temperatureK < 0.0)
    {
        return -1.0;
    }
    const double solidus = coldWallSolidusTemperature(parameters);
    const double liquidus = coldWallLiquidusTemperature(parameters);
    const double hSolidus = parameters.solidSpecificHeatJkgK*solidus;
    if (temperatureK <= solidus)
    {
        return parameters.solidSpecificHeatJkgK*temperatureK;
    }
    const double mushyCp = coldWallMushySpecificHeat(parameters);
    if (temperatureK <= liquidus)
    {
        return hSolidus + mushyCp*(temperatureK - solidus);
    }
    const double hLiquidus = hSolidus + mushyCp*parameters.mushyRangeK;
    return
        hLiquidus
      + coldWallLiquidSensibleIncrement(liquidus, temperatureK);
}

GPU_COLD_WALL_HD inline double coldWallTemperatureFromSpecificEnthalpyK
(
    const double specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (!finiteColdWallValue(specificEnthalpyJkg) || specificEnthalpyJkg < 0.0)
    {
        return -1.0;
    }
    const double solidus = coldWallSolidusTemperature(parameters);
    const double liquidus = coldWallLiquidusTemperature(parameters);
    const double hSolidus = parameters.solidSpecificHeatJkgK*solidus;
    if (specificEnthalpyJkg <= hSolidus)
    {
        return specificEnthalpyJkg/parameters.solidSpecificHeatJkgK;
    }
    const double mushyCp = coldWallMushySpecificHeat(parameters);
    const double hLiquidus = hSolidus + mushyCp*parameters.mushyRangeK;
    if (specificEnthalpyJkg <= hLiquidus)
    {
        return solidus + (specificEnthalpyJkg - hSolidus)/mushyCp;
    }

    double lower = liquidus;
    double upper = liquidus + 1000.0;
    while
    (
        coldWallSpecificEnthalpyJkg(upper, parameters)
      < specificEnthalpyJkg
     && upper < 20000.0
    )
    {
        upper *= 1.5;
    }
    for (int iteration = 0; iteration < 40; ++iteration)
    {
        const double middle = 0.5*(lower + upper);
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
    return 0.5*(lower + upper);
}

GPU_COLD_WALL_HD inline double coldWallSolidFraction
(
    const double specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const double temperatureK = coldWallTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg,
        parameters
    );
    const double solidus = coldWallSolidusTemperature(parameters);
    const double liquidus = coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus)
    {
        return 1.0;
    }
    if (temperatureK >= liquidus)
    {
        return 0.0;
    }
    return (liquidus - temperatureK)/parameters.mushyRangeK;
}

GPU_COLD_WALL_HD inline double coldWallApparentSpecificHeat
(
    const double temperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const double solidus = coldWallSolidusTemperature(parameters);
    const double liquidus = coldWallLiquidusTemperature(parameters);
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

GPU_COLD_WALL_HD inline double coldWallThermalConductivity
(
    const double specificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const double solidFraction = coldWallSolidFraction
    (
        specificEnthalpyJkg,
        parameters
    );
    return
        solidFraction*parameters.solidThermalConductivityWmK
      + (1.0 - solidFraction)*aluminaThermalConductivityWmK;
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
     && parameters.meltingTemperatureK > 0.0
     && parameters.mushyRangeK > 0.0
     && parameters.latentHeatJkg > 0.0
     && parameters.solidDensityKgM3 > 0.0
     && parameters.solidSpecificHeatJkgK > 0.0
     && parameters.solidThermalConductivityWmK > 0.0
     && parameters.pinningThicknessFraction > 0.0
     && parameters.pinningThicknessFraction <= 1.0
     && parameters.interfaceResistanceM2KW >= 0.0
     && (parameters.wallTransientResistance == 0
      || parameters.wallTransientResistance == 1)
     && parameters.nonlinearIterations >= 1
     && parameters.nonlinearIterations <= 12;
}

GPU_COLD_WALL_HD inline void initialiseColdWallProfile
(
    double* nodeSpecificEnthalpyJkg,
    double* ringSolidMassKg,
    const double initialTemperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const double initialEnthalpy = coldWallSpecificEnthalpyJkg
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
        ringSolidMassKg[ring] = 0.0;
    }
}

GPU_COLD_WALL_HD inline void solveColdWallTridiagonal
(
    double* lower,
    double* diagonal,
    double* upper,
    double* rightHandSide,
    double* solution
) noexcept
{
    for (int row = 1; row < coldWallAxialNodeCount; ++row)
    {
        const double multiplier = lower[row]/diagonal[row - 1];
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
    double* nodeSpecificEnthalpyJkg,
    double* ringSolidMassKg,
    double& profileContactAgeS,
    double& frozenFootprintAreaM2,
    const ColdWallSolidificationParameters& parameters,
    const double particleVolumeM3,
    const double parcelMassKg,
    const double maximumAreaM2,
    const double intrinsicContactAreaM2,
    const double contactDurationS,
    const double peakTimeFraction,
    const double deltaTSeconds,
    const double wallTemperatureK,
    const double wallEffusivity,
    const double thermalAreaFactor,
    const double gasTemperatureK,
    const double gasConductanceWK
) noexcept
{
    if
    (
        nodeSpecificEnthalpyJkg == nullptr
     || ringSolidMassKg == nullptr
     || !validColdWallSolidificationParameters(parameters)
     || !(particleVolumeM3 > 0.0)
     || !(parcelMassKg > 0.0)
     || !(maximumAreaM2 > 0.0)
     || !(intrinsicContactAreaM2 > 0.0)
     || !(contactDurationS > 0.0)
     || !(peakTimeFraction > 0.0)
     || !(peakTimeFraction < 1.0)
     || !(deltaTSeconds >= 0.0)
     || !(wallTemperatureK > 0.0)
     || !(wallEffusivity > 0.0)
     || !(thermalAreaFactor > 0.0)
     || thermalAreaFactor > 1.0
     || !(gasTemperatureK > 0.0)
     || gasConductanceWK < 0.0
    )
    {
        return {false, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double ageMid = profileContactAgeS + 0.5*deltaTSeconds;
    const double ageNew = profileContactAgeS + deltaTSeconds;
    const double contactArea = coldWallClamp
    (
        intrinsicContactAreaM2,
        DBL_MIN,
        maximumAreaM2
    );
    const double filmThickness = particleVolumeM3/contactArea;
    const double nodeThickness =
        filmThickness/static_cast<double>(coldWallAxialNodeCount);
    const double nodeMass =
        parcelMassKg/static_cast<double>(coldWallAxialNodeCount);
    const double ringArea =
        maximumAreaM2/static_cast<double>(coldWallRadialRingCount);
    double oldEnthalpy[coldWallAxialNodeCount];
    double guessedTemperature[coldWallAxialNodeCount];
    double candidateEnthalpy[coldWallAxialNodeCount];
    double faceConductance[coldWallAxialNodeCount - 1];
    double ringCoolingPower[coldWallRadialRingCount];
    double ringWetArea[coldWallRadialRingCount];
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

    double wallConductance = 0.0;
    double acceptedWallPower = 0.0;
    for (int iteration = 0; iteration < parameters.nonlinearIterations; ++iteration)
    {
        wallConductance = 0.0;
        const double bottomConductivity = coldWallThermalConductivity
        (
            candidateEnthalpy[0],
            parameters
        );
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const double ringInnerArea = static_cast<double>(ring)*ringArea;
            const double wetArea = coldWallClamp
            (
                contactArea - ringInnerArea,
                0.0,
                ringArea
            );
            ringWetArea[ring] = wetArea;
            if (!(wetArea > 0.0))
            {
                ringCoolingPower[ring] = 0.0;
                continue;
            }
            const double radialFraction =
                (static_cast<double>(ring) + 0.5)
               /static_cast<double>(coldWallRadialRingCount);
            const double spreadCoordinate =
                1.0 - ::sqrt(coldWallClamp(1.0 - radialFraction, 0.0, 1.0));
            const double firstWetAge =
                contactDurationS*peakTimeFraction*spreadCoordinate;
            const double localAge =
                ageMid > firstWetAge ? ageMid - firstWetAge : 0.0;
            const double particleResistance =
                0.5*nodeThickness/bottomConductivity;
            const double wallResistance =
                parameters.wallTransientResistance != 0
              ? ::sqrt(coldWallPi*localAge)/wallEffusivity
              : 0.0;
            const double resistance =
                particleResistance
              + parameters.interfaceResistanceM2KW
              + wallResistance;
            const double conductance =
                thermalAreaFactor*wetArea/(resistance + DBL_MIN);
            wallConductance += conductance;
            ringCoolingPower[ring] = conductance*
                (guessedTemperature[0] - wallTemperatureK);
        }

        for (int face = 0; face < coldWallAxialNodeCount - 1; ++face)
        {
            const double leftK = coldWallThermalConductivity
            (
                candidateEnthalpy[face],
                parameters
            );
            const double rightK = coldWallThermalConductivity
            (
                candidateEnthalpy[face + 1],
                parameters
            );
            const double harmonicK = 2.0*leftK*rightK/(leftK + rightK);
            faceConductance[face] =
                harmonicK*contactArea/nodeThickness;
        }

        double lower[coldWallAxialNodeCount];
        double diagonal[coldWallAxialNodeCount];
        double upper[coldWallAxialNodeCount];
        double rightHandSide[coldWallAxialNodeCount];
        double solvedTemperature[coldWallAxialNodeCount];
        for (int node = 0; node < coldWallAxialNodeCount; ++node)
        {
            lower[node] = 0.0;
            upper[node] = 0.0;
            const double capacity =
                nodeMass*coldWallApparentSpecificHeat
                (
                    guessedTemperature[node],
                    parameters
                )/(deltaTSeconds + DBL_MIN);
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

        double internalPower[coldWallAxialNodeCount - 1];
        for (int face = 0; face < coldWallAxialNodeCount - 1; ++face)
        {
            internalPower[face] = faceConductance[face]
               *(solvedTemperature[face] - solvedTemperature[face + 1]);
        }
        const double wallPower = wallConductance
           *(solvedTemperature[0] - wallTemperatureK);
        acceptedWallPower = wallPower;
        const double gasPower = gasConductanceWK
           *(gasTemperatureK - solvedTemperature[coldWallAxialNodeCount - 1]);
        for (int node = 0; node < coldWallAxialNodeCount; ++node)
        {
            double power = 0.0;
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
             || candidateEnthalpy[node] < 0.0
            )
            {
                return {false, 0.0, 0.0, 0.0, 0.0, 0.0};
            }
            guessedTemperature[node] = coldWallTemperatureFromSpecificEnthalpyK
            (
                candidateEnthalpy[node],
                parameters
            );
        }
    }

    double enthalpySum = 0.0;
    double connectedFraction = 1.0;
    double connectedMass = 0.0;
    for (int node = 0; node < coldWallAxialNodeCount; ++node)
    {
        nodeSpecificEnthalpyJkg[node] = candidateEnthalpy[node];
        enthalpySum += candidateEnthalpy[node];
        const double solidFraction = coldWallSolidFraction
        (
            candidateEnthalpy[node],
            parameters
        );
        connectedFraction =
            solidFraction < connectedFraction ? solidFraction : connectedFraction;
        connectedMass += nodeMass*connectedFraction;
    }

    double assignedMass = 0.0;
    for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
    {
        assignedMass += ringSolidMassKg[ring];
    }
    if (connectedMass < assignedMass && assignedMass > 0.0)
    {
        const double scale = connectedMass/assignedMass;
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            ringSolidMassKg[ring] *= scale;
        }
        assignedMass = connectedMass;
    }
    double remainingMass = connectedMass - assignedMass;
    for
    (
        int pass = 0;
        pass < coldWallRadialRingCount && remainingMass > 1.0e-18*parcelMassKg;
        ++pass
    )
    {
        double weightSum = 0.0;
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const double capacity =
                parameters.solidDensityKgM3*ringWetArea[ring]*filmThickness;
            if (capacity > ringSolidMassKg[ring])
            {
                const double cooling = ringCoolingPower[ring] > 0.0
                  ? ringCoolingPower[ring] : 0.0;
                weightSum += cooling > 0.0 ? cooling : ringWetArea[ring];
            }
        }
        if (!(weightSum > 0.0))
        {
            break;
        }
        double allocatedThisPass = 0.0;
        for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
        {
            const double capacity =
                parameters.solidDensityKgM3*ringWetArea[ring]*filmThickness;
            const double available = capacity - ringSolidMassKg[ring];
            if (!(available > 0.0))
            {
                continue;
            }
            const double cooling = ringCoolingPower[ring] > 0.0
              ? ringCoolingPower[ring] : 0.0;
            const double weight = cooling > 0.0 ? cooling : ringWetArea[ring];
            const double requested = remainingMass*weight/weightSum;
            const double addition = requested < available ? requested : available;
            ringSolidMassKg[ring] += addition;
            allocatedThisPass += addition;
        }
        remainingMass -= allocatedThisPass;
        if (!(allocatedThisPass > 0.0))
        {
            break;
        }
    }

    double candidateFrozenArea = 0.0;
    for (int ring = 0; ring < coldWallRadialRingCount; ++ring)
    {
        if (!(ringWetArea[ring] > 0.0))
        {
            break;
        }
        const double localSolidFraction = ringSolidMassKg[ring]
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

    const double meanEnthalpy =
        enthalpySum/static_cast<double>(coldWallAxialNodeCount);
    const double meanTemperature = coldWallTemperatureFromSpecificEnthalpyK
    (
        meanEnthalpy,
        parameters
    );
    const double surfaceTemperature = coldWallTemperatureFromSpecificEnthalpyK
    (
        candidateEnthalpy[coldWallAxialNodeCount - 1],
        parameters
    );
    const double wallEnergy = acceptedWallPower*deltaTSeconds;
    const bool valid =
        finiteColdWallValue(meanTemperature)
     && finiteColdWallValue(surfaceTemperature)
     && finiteColdWallValue(wallEnergy)
     && finiteColdWallValue(connectedMass)
     && finiteColdWallValue(frozenFootprintAreaM2)
     && meanTemperature > 0.0
     && surfaceTemperature > 0.0
     && connectedMass >= 0.0
     && frozenFootprintAreaM2 >= 0.0;
    return
    {
        valid,
        valid ? meanTemperature : 0.0,
        valid ? surfaceTemperature : 0.0,
        valid ? wallEnergy : 0.0,
        valid ? connectedMass : 0.0,
        valid ? frozenFootprintAreaM2 : 0.0
    };
}

}
}

#undef GPU_COLD_WALL_HD

#endif
