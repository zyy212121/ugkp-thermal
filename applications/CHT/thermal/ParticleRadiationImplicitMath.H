#ifndef GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H
#define GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H

#include "label.H"
#include "scalar.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
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
    if
    (
        !finiteParticleRadiationScalar(acceptedParticleEnergyJ)
     || !finiteParticleRadiationScalar(rawBoundaryEnergyJ)
     || !finiteParticleRadiationScalar(grossRawEnergyJ)
     || !finiteParticleRadiationScalar(relativeTolerance)
     || grossRawEnergyJ < scalar(0)
     || relativeTolerance <= scalar(0)
    )
    {
        implicitRootFailure("invalid conservative radiation-boundary scale inputs");
    }

    const scalar roundoffToleranceJ =
        scalar(4096)*std::numeric_limits<scalar>::epsilon()
       *std::max(grossRawEnergyJ, scalar(1));
    const scalar zeroToleranceJ =
        relativeTolerance*grossRawEnergyJ + roundoffToleranceJ;

    if (std::abs(rawBoundaryEnergyJ) <= zeroToleranceJ)
    {
        const scalar residualJ =
            acceptedParticleEnergyJ + rawBoundaryEnergyJ;
        if (std::abs(residualJ) > zeroToleranceJ)
        {
            implicitRootFailure
            (
                "nonzero particle radiation energy has no representable boundary balance"
            );
        }
        return ConservativeBoundaryEnergyScaleResult{scalar(1), residualJ};
    }

    const scalar scale = -acceptedParticleEnergyJ/rawBoundaryEnergyJ;
    const scalar scaleTolerance =
        relativeTolerance
      + scalar(4096)*std::numeric_limits<scalar>::epsilon();
    if
    (
        !finiteParticleRadiationScalar(scale)
     || scale < scalar(0)
     || scale > scalar(1) + scaleTolerance
    )
    {
        implicitRootFailure
        (
            "implicit radiation update requires a negative or amplifying boundary scale"
        );
    }

    const scalar residualJ =
        acceptedParticleEnergyJ + scale*rawBoundaryEnergyJ;
    const scalar appliedGrossEnergyJ =
        std::abs(acceptedParticleEnergyJ)
      + std::abs(scale*rawBoundaryEnergyJ);
    const scalar conservationToleranceJ =
        relativeTolerance*appliedGrossEnergyJ
      + scalar(4096)*std::numeric_limits<scalar>::epsilon()
       *std::max(appliedGrossEnergyJ, scalar(1));
    if
    (
        !finiteParticleRadiationScalar(residualJ)
     || std::abs(residualJ) > conservationToleranceJ
    )
    {
        implicitRootFailure
        (
            "implicit radiation boundary rescale failed global conservation"
        );
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
     || minimumTemperatureK >= maximumTemperatureK
     || oldTemperatureK < minimumTemperatureK
     || oldTemperatureK > maximumTemperatureK
     || equilibriumTemperatureK < minimumTemperatureK
     || equilibriumTemperatureK > maximumTemperatureK
     || capacityJPerK <= scalar(0)
     || elapsedTimeS <= scalar(0)
     || maxIterations < 1
     || temperatureRelativeTolerance <= scalar(0)
     || energyRelativeTolerance <= scalar(0)
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
        implicitRootFailure("implicit particle-radiation material root has no sign-changing bracket");
    }

    for (label iteration = 1; iteration <= maxIterations; ++iteration)
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
            implicitRootFailure("implicit particle-radiation material root lost its bracket");
        }
    }

    std::ostringstream message;
    message.precision(17);
    message
        << "implicit particle-radiation material root reached its iteration limit"
        << "; oldTemperatureK=" << oldTemperatureK
        << "; equilibriumTemperatureK=" << equilibriumTemperatureK
        << "; lowerTemperatureK=" << lowerTemperatureK
        << "; upperTemperatureK=" << upperTemperatureK
        << "; lowerResidualJ=" << lowerResidualJ
        << "; upperResidualJ=" << upperResidualJ;
    throw ParticleRadiationSolveError(message.str());
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
     || minimumTemperatureK >= maximumTemperatureK
     || oldTemperatureK < minimumTemperatureK
     || oldTemperatureK > maximumTemperatureK
     || maxIterations < 1
     || temperatureRelativeTolerance <= scalar(0)
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
        implicitRootFailure("radiative equilibrium lies outside the legal temperature interval");
    }

    for (label iteration = 1; iteration <= maxIterations; ++iteration)
    {
        const scalar middleTemperatureK =
            scalar(0.5)*(lowerTemperatureK + upperTemperatureK);
        const scalar middleBalance =
            radiativeBalanceAtTemperature(middleTemperatureK);
        if (!finiteParticleRadiationScalar(middleBalance))
        {
            implicitRootFailure("radiative-equilibrium iteration is nonfinite");
        }
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
            implicitRootFailure("radiative-equilibrium root lost its bracket");
        }

        const scalar relativeInterval =
            std::abs(upperTemperatureK - lowerTemperatureK)
           /std::max(std::abs(middleTemperatureK), scalar(1));
        if (relativeInterval <= temperatureRelativeTolerance)
        {
            return scalar(0.5)*(lowerTemperatureK + upperTemperatureK);
        }
    }

    implicitRootFailure("radiative-equilibrium root reached its iteration limit");
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
     || minimumTemperatureK >= maximumTemperatureK
     || oldTemperatureK < minimumTemperatureK
     || oldTemperatureK > maximumTemperatureK
     || maxIterations < 1
     || temperatureRelativeTolerance <= scalar(0)
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
        bool equilibriumBracketConverged = false;
        for (label iteration = 1; iteration <= maxIterations; ++iteration)
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
                equilibriumBracketConverged = true;
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
                implicitRootFailure("bounded radiative-equilibrium bracket was lost");
            }

            const scalar relativeInterval =
                std::abs(farTemperatureK - nearTemperatureK)
               /std::max(std::abs(middleTemperatureK), scalar(1));
            if (relativeInterval <= temperatureRelativeTolerance)
            {
                equilibriumBracketConverged = true;
                break;
            }
        }
        if (!equilibriumBracketConverged)
        {
            implicitRootFailure("bounded radiative-equilibrium bracket reached its iteration limit");
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
