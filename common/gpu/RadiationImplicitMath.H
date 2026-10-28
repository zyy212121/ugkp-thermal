#ifndef UGKP_RADIATION_IMPLICIT_MATH_H
#define UGKP_RADIATION_IMPLICIT_MATH_H
#include <cmath>
#include <limits>
#ifdef __CUDACC__
#define UGKP_RADIATION_HD __host__ __device__
#else
#define UGKP_RADIATION_HD
#endif
namespace ugkp { namespace gpu { namespace radiationMath {
constexpr double radiationDoubleMinimum=std::numeric_limits<double>::min();
constexpr double radiationDoubleEpsilon=std::numeric_limits<double>::epsilon();
// Preserve std::min/max operand order, including unordered IEEE comparisons.
template<class T> UGKP_RADIATION_HD inline T radiationMaximum(T a,T b){return a<b?b:a;}
template<class T> UGKP_RADIATION_HD inline T radiationMinimum(T a,T b){return b<a?b:a;}
struct ImplicitTemperatureRootResult
{
    double temperatureK;
    double residualJ;
    int iterations;
};

struct ConservativeBoundaryEnergyScaleResult
{
    double scale;
    double residualJ;
};

UGKP_RADIATION_HD inline bool finiteParticleRadiationScalar(const double value)
{
    return std::isfinite(value);
}


UGKP_RADIATION_HD inline ConservativeBoundaryEnergyScaleResult conservativeBoundaryEnergyScale
(
    const double acceptedParticleEnergyJ,
    const double rawBoundaryEnergyJ,
    const double grossRawEnergyJ,
    const double relativeTolerance
)
{


    if (finiteParticleRadiationScalar(grossRawEnergyJ)
        && finiteParticleRadiationScalar(relativeTolerance)
        && grossRawEnergyJ >= double(0) && relativeTolerance > double(0))
    {
        const double roundoffToleranceJ =
            double(4096)*radiationDoubleEpsilon
           *radiationMaximum(grossRawEnergyJ, double(1));
        const double zeroToleranceJ =
            relativeTolerance*grossRawEnergyJ + roundoffToleranceJ;
        if (std::abs(rawBoundaryEnergyJ) <= zeroToleranceJ)
        {
            const double residualJ = acceptedParticleEnergyJ + rawBoundaryEnergyJ;
            if (std::abs(residualJ) <= zeroToleranceJ)
            {
                return ConservativeBoundaryEnergyScaleResult{double(1), residualJ};
            }
        }
    }
    if (rawBoundaryEnergyJ == double(0) && acceptedParticleEnergyJ == double(0))
    {
        return ConservativeBoundaryEnergyScaleResult{double(1), double(0)};
    }
    const double scale = -acceptedParticleEnergyJ/rawBoundaryEnergyJ;
    const double residualJ = acceptedParticleEnergyJ + scale*rawBoundaryEnergyJ;

    return ConservativeBoundaryEnergyScaleResult{scale, residualJ};
}

UGKP_RADIATION_HD inline bool oppositeSigns(const double first, const double second)
{
    return
        (first < double(0) && second > double(0))
     || (first > double(0) && second < double(0));
}

template<class PowerAtTemperature>
UGKP_RADIATION_HD ImplicitTemperatureRootResult solveImplicitMaterialTemperature
(
    const double oldTemperatureK,
    const double equilibriumTemperatureK,
    const double minimumTemperatureK,
    const double maximumTemperatureK,
    const double capacityJPerK,
    const double elapsedTimeS,
    const int maxIterations,
    const double temperatureRelativeTolerance,
    const double energyRelativeTolerance,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    const auto residualAt = [&](const double temperatureK)
    {
        const double capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const double radiationEnergyJ =
            elapsedTimeS*powerAtTemperatureW(temperatureK);
        const double residualJ = capacityEnergyJ - radiationEnergyJ;

        return residualJ;
    };

    double lowerTemperatureK = radiationMinimum(oldTemperatureK, equilibriumTemperatureK);
    double upperTemperatureK = radiationMaximum(oldTemperatureK, equilibriumTemperatureK);
    double lowerResidualJ = residualAt(lowerTemperatureK);
    double upperResidualJ = residualAt(upperTemperatureK);

    if (lowerResidualJ == double(0))
    {
        return ImplicitTemperatureRootResult{lowerTemperatureK, double(0), 0};
    }
    if (upperResidualJ == double(0))
    {
        return ImplicitTemperatureRootResult{upperTemperatureK, double(0), 0};
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
        const double temperatureK =
            double(0.5)*(lowerTemperatureK + upperTemperatureK);
        const double powerW = powerAtTemperatureW(temperatureK);

        const double capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const double radiationEnergyJ = elapsedTimeS*powerW;
        const double residualJ = capacityEnergyJ - radiationEnergyJ;

        last = ImplicitTemperatureRootResult{temperatureK, residualJ, iteration};
        const double energyScaleJ = radiationMaximum
        (
            radiationMaximum(std::abs(capacityEnergyJ), std::abs(radiationEnergyJ)),
            radiationDoubleMinimum
        );
        const double relativeResidual = std::abs(residualJ)/energyScaleJ;
        const double relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /radiationMaximum
            (
                radiationMaximum(std::abs(temperatureK), std::abs(oldTemperatureK)),
                double(1)
            );
        const double absoluteRoundoffToleranceJ =
            double(4096)*radiationDoubleEpsilon
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
                    double(1)
                ),
                radiationMaximum(std::abs(radiationEnergyJ), double(1))
            );

        if
        (
            residualJ == double(0)
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
UGKP_RADIATION_HD double solveRadiativeEquilibriumTemperature
(
    const double oldTemperatureK,
    const double minimumTemperatureK,
    const double maximumTemperatureK,
    const int maxIterations,
    const double temperatureRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature
)
{
    const double oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);

    if (oldBalance == double(0))
    {
        return oldTemperatureK;
    }

    double lowerTemperatureK = oldBalance > double(0)
      ? oldTemperatureK
      : minimumTemperatureK;
    double upperTemperatureK = oldBalance > double(0)
      ? maximumTemperatureK
      : oldTemperatureK;
    double lowerBalance = radiativeBalanceAtTemperature(lowerTemperatureK);
    double upperBalance = radiativeBalanceAtTemperature(upperTemperatureK);

    if (lowerBalance == double(0))
    {
        return lowerTemperatureK;
    }
    if (upperBalance == double(0))
    {
        return upperTemperatureK;
    }
    if (!oppositeSigns(lowerBalance, upperBalance))
    {
        return std::abs(lowerBalance) <= std::abs(upperBalance)
          ? lowerTemperatureK : upperTemperatureK;
    }

    double lastTemperatureK = oldTemperatureK;
    for (int iteration = 1; iteration <= radiationMaximum(maxIterations, int(1)); ++iteration)
    {
        const double middleTemperatureK =
            double(0.5)*(lowerTemperatureK + upperTemperatureK);
        const double middleBalance =
            radiativeBalanceAtTemperature(middleTemperatureK);

        lastTemperatureK = middleTemperatureK;
        if (middleBalance == double(0))
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

        const double relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /radiationMaximum(std::abs(middleTemperatureK), double(1));
        if (relativeInterval <= temperatureRelativeTolerance)
        {
            return double(0.5)*(lowerTemperatureK + upperTemperatureK);
        }
    }

    return lastTemperatureK;
}

template<class RadiativeBalanceAtTemperature, class PowerAtTemperature>
UGKP_RADIATION_HD ImplicitTemperatureRootResult solveBoundedImplicitMaterialTemperature
(
    const double oldTemperatureK,
    const double minimumTemperatureK,
    const double maximumTemperatureK,
    const double capacityJPerK,
    const double elapsedTimeS,
    const int maxIterations,
    const double temperatureRelativeTolerance,
    const double energyRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    const double oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);

    if (oldBalance == double(0))
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

    double nearTemperatureK = oldTemperatureK;
    double nearBalance = oldBalance;
    double farTemperatureK = oldBalance > double(0)
      ? maximumTemperatureK
      : minimumTemperatureK;
    double farBalance = radiativeBalanceAtTemperature(farTemperatureK);

    if (farBalance != double(0) && oppositeSigns(nearBalance, farBalance))
    {
        for (int iteration = 1; iteration <= radiationMaximum(maxIterations, int(1)); ++iteration)
        {
            const double middleTemperatureK =
                double(0.5)*(nearTemperatureK + farTemperatureK);
            const double middleBalance =
                radiativeBalanceAtTemperature(middleTemperatureK);

            if (middleBalance == double(0))
            {
                farTemperatureK = middleTemperatureK;
                farBalance = double(0);
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

            const double relativeInterval =
                std::abs(farTemperatureK - nearTemperatureK)
               /radiationMaximum(std::abs(middleTemperatureK), double(1));
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
