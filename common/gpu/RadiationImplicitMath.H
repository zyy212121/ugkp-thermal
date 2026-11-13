#ifndef UGKP_RADIATION_IMPLICIT_MATH_H
#define UGKP_RADIATION_IMPLICIT_MATH_H
#include "RadiationPrecision.H"
#include <cmath>
#include <limits>
#ifdef __CUDACC__
#define UGKP_RADIATION_HD __host__ __device__
#else
#define UGKP_RADIATION_HD
#endif
namespace ugkp { namespace gpu { namespace radiationMath {
constexpr RadiationReal radiationDoubleMinimum=std::numeric_limits<RadiationReal>::min();
constexpr RadiationReal radiationDoubleEpsilon=std::numeric_limits<RadiationReal>::epsilon();
// Preserve std::min/max operand order, including unordered IEEE comparisons.
template<class T> UGKP_RADIATION_HD inline T radiationMaximum(T a,T b){return a<b?b:a;}
template<class T> UGKP_RADIATION_HD inline T radiationMinimum(T a,T b){return b<a?b:a;}
struct ImplicitTemperatureRootResult
{
    RadiationReal temperatureK;
    RadiationReal residualJ;
    int iterations;
};

struct ConservativeBoundaryEnergyScaleResult
{
    RadiationReal scale;
    RadiationReal residualJ;
};

UGKP_RADIATION_HD inline bool finiteParticleRadiationScalar(const RadiationReal value)
{
    return std::isfinite(value);
}


UGKP_RADIATION_HD inline ConservativeBoundaryEnergyScaleResult conservativeBoundaryEnergyScale
(
    const RadiationReal acceptedParticleEnergyJ,
    const RadiationReal rawBoundaryEnergyJ,
    const RadiationReal grossRawEnergyJ,
    const RadiationReal relativeTolerance,
    const RadiationReal radiativeEnergyMagnitudeJ = 0
)
{


    if (finiteParticleRadiationScalar(grossRawEnergyJ)
        && finiteParticleRadiationScalar(relativeTolerance)
        && grossRawEnergyJ >= RadiationReal(0) && relativeTolerance > RadiationReal(0))
    {
        const RadiationReal roundoffToleranceJ =
            RadiationReal(4096)*radiationDoubleEpsilon
           *radiationMaximum(radiationMaximum(grossRawEnergyJ, radiativeEnergyMagnitudeJ), RadiationReal(1));
        const RadiationReal zeroToleranceJ =
            relativeTolerance*grossRawEnergyJ + roundoffToleranceJ;
        if (std::abs(rawBoundaryEnergyJ) <= zeroToleranceJ)
        {
            const RadiationReal residualJ = acceptedParticleEnergyJ + rawBoundaryEnergyJ;
            if (std::abs(residualJ) <= zeroToleranceJ)
            {
                return ConservativeBoundaryEnergyScaleResult{RadiationReal(1), residualJ};
            }
        }
    }
    if (rawBoundaryEnergyJ == RadiationReal(0) && acceptedParticleEnergyJ == RadiationReal(0))
    {
        return ConservativeBoundaryEnergyScaleResult{RadiationReal(1), RadiationReal(0)};
    }
    const RadiationReal scale = -acceptedParticleEnergyJ/rawBoundaryEnergyJ;
    const RadiationReal residualJ = acceptedParticleEnergyJ + scale*rawBoundaryEnergyJ;

    return ConservativeBoundaryEnergyScaleResult{scale, residualJ};
}

UGKP_RADIATION_HD inline bool oppositeSigns(const RadiationReal first, const RadiationReal second)
{
    return
        (first < RadiationReal(0) && second > RadiationReal(0))
     || (first > RadiationReal(0) && second < RadiationReal(0));
}

template<class PowerAtTemperature>
UGKP_RADIATION_HD ImplicitTemperatureRootResult solveImplicitMaterialTemperature
(
    const RadiationReal oldTemperatureK,
    const RadiationReal equilibriumTemperatureK,
    const RadiationReal minimumTemperatureK,
    const RadiationReal maximumTemperatureK,
    const RadiationReal capacityJPerK,
    const RadiationReal elapsedTimeS,
    const int maxIterations,
    const RadiationReal temperatureRelativeTolerance,
    const RadiationReal energyRelativeTolerance,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    const auto residualAt = [&](const RadiationReal temperatureK)
    {
        const RadiationReal capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const RadiationReal radiationEnergyJ =
            elapsedTimeS*powerAtTemperatureW(temperatureK);
        const RadiationReal residualJ = capacityEnergyJ - radiationEnergyJ;

        return residualJ;
    };

    RadiationReal lowerTemperatureK = radiationMinimum(oldTemperatureK, equilibriumTemperatureK);
    RadiationReal upperTemperatureK = radiationMaximum(oldTemperatureK, equilibriumTemperatureK);
    RadiationReal lowerResidualJ = residualAt(lowerTemperatureK);
    RadiationReal upperResidualJ = residualAt(upperTemperatureK);

    if (lowerResidualJ == RadiationReal(0))
    {
        return ImplicitTemperatureRootResult{lowerTemperatureK, RadiationReal(0), 0};
    }
    if (upperResidualJ == RadiationReal(0))
    {
        return ImplicitTemperatureRootResult{upperTemperatureK, RadiationReal(0), 0};
    }
    if (!oppositeSigns(lowerResidualJ, upperResidualJ))
    {
        return std::abs(lowerResidualJ) <= std::abs(upperResidualJ)
          ? ImplicitTemperatureRootResult{lowerTemperatureK, lowerResidualJ, 0}
          : ImplicitTemperatureRootResult{upperTemperatureK, upperResidualJ, 0};
    }
    ImplicitTemperatureRootResult last{lowerTemperatureK, lowerResidualJ, 0};

    for (int iteration = 1; iteration <= radiationMaximum(maxIterations, int(1)); ++iteration)
    {
        const RadiationReal temperatureK =
            RadiationReal(0.5)*(lowerTemperatureK + upperTemperatureK);
        const RadiationReal powerW = powerAtTemperatureW(temperatureK);

        const RadiationReal capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const RadiationReal radiationEnergyJ = elapsedTimeS*powerW;
        const RadiationReal residualJ = capacityEnergyJ - radiationEnergyJ;

        last = ImplicitTemperatureRootResult{temperatureK, residualJ, iteration};
        const RadiationReal energyScaleJ = radiationMaximum
        (
            radiationMaximum(std::abs(capacityEnergyJ), std::abs(radiationEnergyJ)),
            radiationDoubleMinimum
        );
        const RadiationReal relativeResidual = std::abs(residualJ)/energyScaleJ;
        const RadiationReal relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /radiationMaximum
            (
                radiationMaximum(std::abs(temperatureK), std::abs(oldTemperatureK)),
                RadiationReal(1)
            );
        const RadiationReal absoluteRoundoffToleranceJ =
            RadiationReal(4096)*radiationDoubleEpsilon
           *radiationMaximum
            (
                std::abs(capacityJPerK)
               *radiationMaximum
                (
                    radiationMaximum
                    (
                        std::abs(oldTemperatureK),
                        std::abs(temperatureK)
                    ),
                    RadiationReal(1)
                ),
                radiationMaximum(std::abs(radiationEnergyJ), RadiationReal(1))
            );

        if
        (
            residualJ == RadiationReal(0)
         ||
            (
                relativeInterval <= temperatureRelativeTolerance
             &&
                (
                    relativeResidual <= energyRelativeTolerance
                 || std::abs(residualJ) <= absoluteRoundoffToleranceJ
                )
            )
        )
        {
            return ImplicitTemperatureRootResult
            {
                temperatureK,
                residualJ,
                iteration
            };
        }

        if (oppositeSigns(lowerResidualJ, residualJ))
        {
            upperTemperatureK = temperatureK;
            upperResidualJ = residualJ;
        }
        else if (oppositeSigns(residualJ, upperResidualJ))
        {
            lowerTemperatureK = temperatureK;
            lowerResidualJ = residualJ;
        }
        else
        {
            return last;
        }
    }

    return last;
}

template<class RadiativeBalanceAtTemperature>
UGKP_RADIATION_HD RadiationReal solveRadiativeEquilibriumTemperature
(
    const RadiationReal oldTemperatureK,
    const RadiationReal minimumTemperatureK,
    const RadiationReal maximumTemperatureK,
    const int maxIterations,
    const RadiationReal temperatureRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature
)
{
    const RadiationReal oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);

    if (oldBalance == RadiationReal(0))
    {
        return oldTemperatureK;
    }

    RadiationReal lowerTemperatureK = oldBalance > RadiationReal(0)
      ? oldTemperatureK
      : minimumTemperatureK;
    RadiationReal upperTemperatureK = oldBalance > RadiationReal(0)
      ? maximumTemperatureK
      : oldTemperatureK;
    RadiationReal lowerBalance = radiativeBalanceAtTemperature(lowerTemperatureK);
    RadiationReal upperBalance = radiativeBalanceAtTemperature(upperTemperatureK);

    if (lowerBalance == RadiationReal(0))
    {
        return lowerTemperatureK;
    }
    if (upperBalance == RadiationReal(0))
    {
        return upperTemperatureK;
    }
    if (!oppositeSigns(lowerBalance, upperBalance))
    {
        return std::abs(lowerBalance) <= std::abs(upperBalance)
          ? lowerTemperatureK : upperTemperatureK;
    }

    RadiationReal lastTemperatureK = oldTemperatureK;
    for (int iteration = 1; iteration <= radiationMaximum(maxIterations, int(1)); ++iteration)
    {
        const RadiationReal middleTemperatureK =
            RadiationReal(0.5)*(lowerTemperatureK + upperTemperatureK);
        const RadiationReal middleBalance =
            radiativeBalanceAtTemperature(middleTemperatureK);

        lastTemperatureK = middleTemperatureK;
        if (middleBalance == RadiationReal(0))
        {
            return middleTemperatureK;
        }

        if (oppositeSigns(lowerBalance, middleBalance))
        {
            upperTemperatureK = middleTemperatureK;
            upperBalance = middleBalance;
        }
        else if (oppositeSigns(middleBalance, upperBalance))
        {
            lowerTemperatureK = middleTemperatureK;
            lowerBalance = middleBalance;
        }
        else
        {
            return lastTemperatureK;
        }

        const RadiationReal relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /radiationMaximum(std::abs(middleTemperatureK), RadiationReal(1));
        if (relativeInterval <= temperatureRelativeTolerance)
        {
            return RadiationReal(0.5)*(lowerTemperatureK + upperTemperatureK);
        }
    }

    return lastTemperatureK;
}

template<class RadiativeBalanceAtTemperature, class PowerAtTemperature>
UGKP_RADIATION_HD ImplicitTemperatureRootResult solveBoundedImplicitMaterialTemperature
(
    const RadiationReal oldTemperatureK,
    const RadiationReal minimumTemperatureK,
    const RadiationReal maximumTemperatureK,
    const RadiationReal capacityJPerK,
    const RadiationReal elapsedTimeS,
    const int maxIterations,
    const RadiationReal temperatureRelativeTolerance,
    const RadiationReal energyRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    const RadiationReal oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);

    if (oldBalance == RadiationReal(0))
    {
        return solveImplicitMaterialTemperature
        (
            oldTemperatureK,
            oldTemperatureK,
            minimumTemperatureK,
            maximumTemperatureK,
            capacityJPerK,
            elapsedTimeS,
            maxIterations,
            temperatureRelativeTolerance,
            energyRelativeTolerance,
            powerAtTemperatureW
        );
    }

    RadiationReal nearTemperatureK = oldTemperatureK;
    RadiationReal nearBalance = oldBalance;
    RadiationReal farTemperatureK = oldBalance > RadiationReal(0)
      ? maximumTemperatureK
      : minimumTemperatureK;
    RadiationReal farBalance = radiativeBalanceAtTemperature(farTemperatureK);

    if (farBalance != RadiationReal(0) && oppositeSigns(nearBalance, farBalance))
    {
        for (int iteration = 1; iteration <= radiationMaximum(maxIterations, int(1)); ++iteration)
        {
            const RadiationReal middleTemperatureK =
                RadiationReal(0.5)*(nearTemperatureK + farTemperatureK);
            const RadiationReal middleBalance =
                radiativeBalanceAtTemperature(middleTemperatureK);

            if (middleBalance == RadiationReal(0))
            {
                farTemperatureK = middleTemperatureK;
                farBalance = RadiationReal(0);
                break;
            }
            if (oppositeSigns(nearBalance, middleBalance))
            {
                farTemperatureK = middleTemperatureK;
                farBalance = middleBalance;
            }
            else if (oppositeSigns(middleBalance, farBalance))
            {
                nearTemperatureK = middleTemperatureK;
                nearBalance = middleBalance;
            }
            else
            {
                break;
            }

            const RadiationReal relativeInterval =
                std::abs(farTemperatureK - nearTemperatureK)
               /radiationMaximum(std::abs(middleTemperatureK), RadiationReal(1));
            if (relativeInterval <= temperatureRelativeTolerance)
            {
                break;
            }
        }
    }

    return solveImplicitMaterialTemperature
    (
        oldTemperatureK,
        farTemperatureK,
        minimumTemperatureK,
        maximumTemperatureK,
        capacityJPerK,
        elapsedTimeS,
        maxIterations,
        temperatureRelativeTolerance,
        energyRelativeTolerance,
        powerAtTemperatureW
    );
}

}}}
#undef UGKP_RADIATION_HD
#endif
