#ifndef GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H
#define GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H

#include "label.H"
#include "scalar.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Foam
{
namespace gpuThermal
{

class ParticleRadiationSolveError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

namespace detail
{

struct ImplicitTemperatureRootResult
{
    scalar temperatureK;
    scalar residualJ;
    label iterations;
};

struct ConservativeBoundaryEnergyScaleResult
{
    scalar scale;
    scalar residualJ;
};

inline bool finiteParticleRadiationScalar(const scalar value)
{
    return std::isfinite(value);
}

[[noreturn]] inline void implicitRootFailure(const char* reason)
{
    throw ParticleRadiationSolveError(reason);
}

inline ConservativeBoundaryEnergyScaleResult conservativeBoundaryEnergyScale
(
    const scalar acceptedParticleEnergyJ,
    const scalar rawBoundaryEnergyJ,
    const scalar grossRawEnergyJ,
    const scalar relativeTolerance
)
{
    if (!finiteParticleRadiationScalar(acceptedParticleEnergyJ)
        || !finiteParticleRadiationScalar(rawBoundaryEnergyJ))
    {
        implicitRootFailure("radiation-boundary energy is nonfinite");
    }
    // Preserve the original successful near-zero identity. Inputs outside its
    // former admitted domain continue to the actual nonzero division below.
    if (finiteParticleRadiationScalar(grossRawEnergyJ)
        && finiteParticleRadiationScalar(relativeTolerance)
        && grossRawEnergyJ >= scalar(0) && relativeTolerance > scalar(0))
    {
        const scalar roundoffToleranceJ =
            scalar(4096)*std::numeric_limits<scalar>::epsilon()
           *std::max(grossRawEnergyJ, scalar(1));
        const scalar zeroToleranceJ =
            relativeTolerance*grossRawEnergyJ + roundoffToleranceJ;
        if (std::abs(rawBoundaryEnergyJ) <= zeroToleranceJ)
        {
            const scalar residualJ = acceptedParticleEnergyJ + rawBoundaryEnergyJ;
            if (std::abs(residualJ) <= zeroToleranceJ)
            {
                return ConservativeBoundaryEnergyScaleResult{scalar(1), residualJ};
            }
        }
    }
    if (rawBoundaryEnergyJ == scalar(0))
    {
        if (acceptedParticleEnergyJ != scalar(0))
        {
            implicitRootFailure("nonzero particle radiation energy has zero boundary divisor");
        }
        return ConservativeBoundaryEnergyScaleResult{scalar(1), scalar(0)};
    }
    const scalar scale = -acceptedParticleEnergyJ/rawBoundaryEnergyJ;
    const scalar residualJ = acceptedParticleEnergyJ + scale*rawBoundaryEnergyJ;
    if (!finiteParticleRadiationScalar(scale) || !finiteParticleRadiationScalar(residualJ))
    {
        implicitRootFailure("radiation-boundary rescale arithmetic is nonfinite");
    }
    return ConservativeBoundaryEnergyScaleResult{scale, residualJ};
}

inline bool oppositeSigns(const scalar first, const scalar second)
{
    return
        (first < scalar(0) && second > scalar(0))
     || (first > scalar(0) && second < scalar(0));
}

template<class PowerAtTemperature>
ImplicitTemperatureRootResult solveImplicitMaterialTemperature
(
    const scalar oldTemperatureK,
    const scalar equilibriumTemperatureK,
    const scalar minimumTemperatureK,
    const scalar maximumTemperatureK,
    const scalar capacityJPerK,
    const scalar elapsedTimeS,
    const label maxIterations,
    const scalar temperatureRelativeTolerance,
    const scalar energyRelativeTolerance,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    if
    (
        !finiteParticleRadiationScalar(oldTemperatureK)
     || !finiteParticleRadiationScalar(equilibriumTemperatureK)
     || !finiteParticleRadiationScalar(minimumTemperatureK)
     || !finiteParticleRadiationScalar(maximumTemperatureK)
     || !finiteParticleRadiationScalar(capacityJPerK)
     || !finiteParticleRadiationScalar(elapsedTimeS)
     || !finiteParticleRadiationScalar(temperatureRelativeTolerance)
     || !finiteParticleRadiationScalar(energyRelativeTolerance)
    )
    {
        implicitRootFailure("invalid implicit particle-radiation root inputs");
    }

    const auto residualAt = [&](const scalar temperatureK)
    {
        const scalar capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const scalar radiationEnergyJ =
            elapsedTimeS*powerAtTemperatureW(temperatureK);
        const scalar residualJ = capacityEnergyJ - radiationEnergyJ;
        if
        (
            !finiteParticleRadiationScalar(capacityEnergyJ)
         || !finiteParticleRadiationScalar(radiationEnergyJ)
         || !finiteParticleRadiationScalar(residualJ)
        )
        {
            implicitRootFailure("implicit particle-radiation residual is nonfinite");
        }
        return residualJ;
    };

    scalar lowerTemperatureK = std::min(oldTemperatureK, equilibriumTemperatureK);
    scalar upperTemperatureK = std::max(oldTemperatureK, equilibriumTemperatureK);
    scalar lowerResidualJ = residualAt(lowerTemperatureK);
    scalar upperResidualJ = residualAt(upperTemperatureK);

    if (lowerResidualJ == scalar(0))
    {
        return ImplicitTemperatureRootResult{lowerTemperatureK, scalar(0), 0};
    }
    if (upperResidualJ == scalar(0))
    {
        return ImplicitTemperatureRootResult{upperTemperatureK, scalar(0), 0};
    }
    if (!oppositeSigns(lowerResidualJ, upperResidualJ))
    {
        return std::abs(lowerResidualJ) <= std::abs(upperResidualJ)
          ? ImplicitTemperatureRootResult{lowerTemperatureK, lowerResidualJ, 0}
          : ImplicitTemperatureRootResult{upperTemperatureK, upperResidualJ, 0};
    }
    ImplicitTemperatureRootResult last{lowerTemperatureK, lowerResidualJ, 0};

    for (label iteration = 1; iteration <= std::max(maxIterations, label(1)); ++iteration)
    {
        const scalar temperatureK =
            scalar(0.5)*(lowerTemperatureK + upperTemperatureK);
        const scalar powerW = powerAtTemperatureW(temperatureK);
        if (!finiteParticleRadiationScalar(powerW))
        {
            implicitRootFailure("implicit particle-radiation power law is nonfinite");
        }
        const scalar capacityEnergyJ =
            capacityJPerK*(temperatureK - oldTemperatureK);
        const scalar radiationEnergyJ = elapsedTimeS*powerW;
        const scalar residualJ = capacityEnergyJ - radiationEnergyJ;
        if (!finiteParticleRadiationScalar(capacityEnergyJ)
            || !finiteParticleRadiationScalar(radiationEnergyJ)
            || !finiteParticleRadiationScalar(residualJ))
        {
            implicitRootFailure("implicit particle-radiation residual is nonfinite");
        }
        last = ImplicitTemperatureRootResult{temperatureK, residualJ, iteration};
        const scalar energyScaleJ = std::max
        (
            std::max(std::abs(capacityEnergyJ), std::abs(radiationEnergyJ)),
            std::numeric_limits<scalar>::min()
        );
        const scalar relativeResidual = std::abs(residualJ)/energyScaleJ;
        const scalar relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /std::max
            (
                std::max(std::abs(temperatureK), std::abs(oldTemperatureK)),
                scalar(1)
            );
        const scalar absoluteRoundoffToleranceJ =
            scalar(4096)*std::numeric_limits<scalar>::epsilon()
           *std::max
            (
                std::abs(capacityJPerK)
               *std::max
                (
                    std::max
                    (
                        std::abs(oldTemperatureK),
                        std::abs(temperatureK)
                    ),
                    scalar(1)
                ),
                std::max(std::abs(radiationEnergyJ), scalar(1))
            );

        if
        (
            residualJ == scalar(0)
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
scalar solveRadiativeEquilibriumTemperature
(
    const scalar oldTemperatureK,
    const scalar minimumTemperatureK,
    const scalar maximumTemperatureK,
    const label maxIterations,
    const scalar temperatureRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature
)
{
    if
    (
        !finiteParticleRadiationScalar(oldTemperatureK)
     || !finiteParticleRadiationScalar(minimumTemperatureK)
     || !finiteParticleRadiationScalar(maximumTemperatureK)
     || !finiteParticleRadiationScalar(temperatureRelativeTolerance)
    )
    {
        implicitRootFailure("invalid radiative-equilibrium root inputs");
    }

    const scalar oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);
    if (!finiteParticleRadiationScalar(oldBalance))
    {
        implicitRootFailure("radiative-equilibrium balance is nonfinite");
    }
    if (oldBalance == scalar(0))
    {
        return oldTemperatureK;
    }

    scalar lowerTemperatureK = oldBalance > scalar(0)
      ? oldTemperatureK
      : minimumTemperatureK;
    scalar upperTemperatureK = oldBalance > scalar(0)
      ? maximumTemperatureK
      : oldTemperatureK;
    scalar lowerBalance = radiativeBalanceAtTemperature(lowerTemperatureK);
    scalar upperBalance = radiativeBalanceAtTemperature(upperTemperatureK);

    if
    (
        !finiteParticleRadiationScalar(lowerBalance)
     || !finiteParticleRadiationScalar(upperBalance)
    )
    {
        implicitRootFailure("radiative-equilibrium bracket balance is nonfinite");
    }
    if (lowerBalance == scalar(0))
    {
        return lowerTemperatureK;
    }
    if (upperBalance == scalar(0))
    {
        return upperTemperatureK;
    }
    if (!oppositeSigns(lowerBalance, upperBalance))
    {
        return std::abs(lowerBalance) <= std::abs(upperBalance)
          ? lowerTemperatureK : upperTemperatureK;
    }

    scalar lastTemperatureK = oldTemperatureK;
    for (label iteration = 1; iteration <= std::max(maxIterations, label(1)); ++iteration)
    {
        const scalar middleTemperatureK =
            scalar(0.5)*(lowerTemperatureK + upperTemperatureK);
        const scalar middleBalance =
            radiativeBalanceAtTemperature(middleTemperatureK);
        if (!finiteParticleRadiationScalar(middleBalance))
        {
            implicitRootFailure("radiative-equilibrium iteration is nonfinite");
        }
        lastTemperatureK = middleTemperatureK;
        if (middleBalance == scalar(0))
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

        const scalar relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /std::max(std::abs(middleTemperatureK), scalar(1));
        if (relativeInterval <= temperatureRelativeTolerance)
        {
            return scalar(0.5)*(lowerTemperatureK + upperTemperatureK);
        }
    }

    return lastTemperatureK;
}

template<class RadiativeBalanceAtTemperature, class PowerAtTemperature>
ImplicitTemperatureRootResult solveBoundedImplicitMaterialTemperature
(
    const scalar oldTemperatureK,
    const scalar minimumTemperatureK,
    const scalar maximumTemperatureK,
    const scalar capacityJPerK,
    const scalar elapsedTimeS,
    const label maxIterations,
    const scalar temperatureRelativeTolerance,
    const scalar energyRelativeTolerance,
    const RadiativeBalanceAtTemperature& radiativeBalanceAtTemperature,
    const PowerAtTemperature& powerAtTemperatureW
)
{
    if
    (
        !finiteParticleRadiationScalar(oldTemperatureK)
     || !finiteParticleRadiationScalar(minimumTemperatureK)
     || !finiteParticleRadiationScalar(maximumTemperatureK)
     || !finiteParticleRadiationScalar(temperatureRelativeTolerance)
    )
    {
        implicitRootFailure("invalid bounded particle-radiation root inputs");
    }

    const scalar oldBalance = radiativeBalanceAtTemperature(oldTemperatureK);
    if (!finiteParticleRadiationScalar(oldBalance))
    {
        implicitRootFailure("bounded particle-radiation balance is nonfinite");
    }
    if (oldBalance == scalar(0))
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

    scalar nearTemperatureK = oldTemperatureK;
    scalar nearBalance = oldBalance;
    scalar farTemperatureK = oldBalance > scalar(0)
      ? maximumTemperatureK
      : minimumTemperatureK;
    scalar farBalance = radiativeBalanceAtTemperature(farTemperatureK);
    if (!finiteParticleRadiationScalar(farBalance))
    {
        implicitRootFailure("bounded particle-radiation endpoint balance is nonfinite");
    }

    if (farBalance != scalar(0) && oppositeSigns(nearBalance, farBalance))
    {
        for (label iteration = 1; iteration <= std::max(maxIterations, label(1)); ++iteration)
        {
            const scalar middleTemperatureK =
                scalar(0.5)*(nearTemperatureK + farTemperatureK);
            const scalar middleBalance =
                radiativeBalanceAtTemperature(middleTemperatureK);
            if (!finiteParticleRadiationScalar(middleBalance))
            {
                implicitRootFailure("bounded equilibrium iteration is nonfinite");
            }
            if (middleBalance == scalar(0))
            {
                farTemperatureK = middleTemperatureK;
                farBalance = scalar(0);
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

            const scalar relativeInterval =
                std::abs(farTemperatureK - nearTemperatureK)
               /std::max(std::abs(middleTemperatureK), scalar(1));
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

}                    
}                        
}                  

#endif
