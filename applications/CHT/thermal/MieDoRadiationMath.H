#ifndef GPU_THERMAL_MIE_DO_RADIATION_MATH_H
#define GPU_THERMAL_MIE_DO_RADIATION_MATH_H

#include "List.H"
#include "label.H"
#include "scalar.H"
#include "scalarField.H"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Foam
{
namespace gpuThermal
{

inline scalar stefanBoltzmannConstant()
{
    return scalar(5.670374419e-8);
}

inline scalar fourPiConstant()
{
    return scalar(12.56637061435917295385057353311801153679);
}

inline scalar bandBlackBodyIntensity
(
    const scalar temperatureK,
    const scalar planckBandFraction
)
{
    const scalar temperatureSquared = temperatureK*temperatureK;
    return
        planckBandFraction*stefanBoltzmannConstant()
       *temperatureSquared*temperatureSquared
       /(fourPiConstant()/scalar(4));
}

inline scalar particleRadiationPowerDensity
(
    const scalar absorptionCoefficient,
    const scalar incidentRadiationG,
    const scalar bandBlackBodyIntensityValue
)
{
    if (absorptionCoefficient == scalar(0))
    {
        return scalar(0);
    }

    return absorptionCoefficient
       *(incidentRadiationG - fourPiConstant()*bandBlackBodyIntensityValue);
}

struct PhaseBalanceMetrics
{
    scalar maxRowError;
    scalar maxWeightedColumnRelativeError;
    scalar maxRelativeScalingCorrection;
    label iterations;
};

struct PhaseBalanceResult
{
    List<scalarField> probabilities;
    PhaseBalanceMetrics metrics;
};

struct InternalFaceUpwindCoefficients
{
    scalar ownerDiag;
    scalar neighbourDiag;
    scalar upper;
    scalar lower;
};

struct RadiationConservationAssessment
{
    scalar residualW;
    scalar netScaleW;
    scalar numericalBoundW;
    scalar relativeResidual;
    bool usedCancellationBound;
    bool accepted;
};

inline RadiationConservationAssessment assessRadiationConservation
(
    const scalar particlePowerW,
    const scalar boundaryPowerW,
    const scalar grossTransportPowerW,
    const scalar finalSourceResidual,
    const scalar relativeTolerance
)
{
    if
    (
        !std::isfinite(particlePowerW)
     || !std::isfinite(boundaryPowerW)
     || !std::isfinite(grossTransportPowerW)
     || grossTransportPowerW < scalar(0)
     || !std::isfinite(finalSourceResidual)
     || finalSourceResidual < scalar(0)
     || !std::isfinite(relativeTolerance)
     || relativeTolerance <= scalar(0)
    )
    {
        throw std::runtime_error("radiation conservation inputs are invalid");
    }

    RadiationConservationAssessment result;
    result.residualW = particlePowerW + boundaryPowerW;
    result.netScaleW = max(mag(particlePowerW), mag(boundaryPowerW));
    result.numericalBoundW =
        (
            scalar(2048)*std::numeric_limits<scalar>::epsilon()
          + scalar(8)*finalSourceResidual
        )
       *max(grossTransportPowerW, scalar(1));
    result.relativeResidual =
        mag(result.residualW)/max(result.netScaleW, SMALL);
    result.usedCancellationBound = result.netScaleW <= result.numericalBoundW;
    result.accepted = result.usedCancellationBound
      ? mag(result.residualW) <= result.numericalBoundW
      : result.relativeResidual <= relativeTolerance;
    return result;
}

inline InternalFaceUpwindCoefficients internalFaceUpwindCoefficients
(
    const scalar orientedAreaFlux
)
{
    InternalFaceUpwindCoefficients result{scalar(0), scalar(0), scalar(0), scalar(0)};
    if (orientedAreaFlux > scalar(0))
    {
        result.ownerDiag = orientedAreaFlux;
        result.lower = -orientedAreaFlux;
    }
    else if (orientedAreaFlux < scalar(0))
    {
        result.neighbourDiag = -orientedAreaFlux;
        result.upper = orientedAreaFlux;
    }
    return result;
}

inline PhaseBalanceMetrics measurePhaseBalance
(
    const List<scalarField>& probabilities,
    const scalarField& weights
)
{
    const label n = probabilities.size();
    if (!n || weights.size() < n)
    {
        throw std::runtime_error("phase probability/weight sizes are inconsistent");
    }

    PhaseBalanceMetrics metrics{scalar(0), scalar(0), scalar(0), 0};

    for (label rowI = 0; rowI < n; ++rowI)
    {
        if (probabilities[rowI].size() < n)
        {
            throw std::runtime_error("phase probability matrix is not square");
        }

        scalar rowSum = 0;
        for (label colI = 0; colI < n; ++colI)
        {
            const scalar value = probabilities[rowI][colI];
            if (!std::isfinite(value))
            {
                throw std::runtime_error("phase probability is nonfinite");
            }
            rowSum += value;
        }
        metrics.maxRowError = max(metrics.maxRowError, mag(rowSum - scalar(1)));
    }

    for (label colI = 0; colI < n; ++colI)
    {
        if (!std::isfinite(weights[colI]) || weights[colI] == scalar(0))
        {
            throw std::runtime_error("phase quadrature weight is zero or nonfinite");
        }

        scalar weightedColumn = 0;
        for (label rowI = 0; rowI < n; ++rowI)
        {
            weightedColumn += weights[rowI]*probabilities[rowI][colI];
        }
        metrics.maxWeightedColumnRelativeError = max
        (
            metrics.maxWeightedColumnRelativeError,
            mag(weightedColumn - weights[colI])/mag(weights[colI])
        );
    }

    return metrics;
}

inline PhaseBalanceMetrics balanceSymmetricPhaseKernelInPlace
(
    const List<scalarField>& rawSymmetricKernel,
    const scalarField& weights,
    const scalar tolerance,
    const label maxIterations,
    scalarField& scaling,
    scalarField& newScaling,
    List<scalarField>& probabilities
)
{
    const label n = rawSymmetricKernel.size();
    if (!n || weights.size() < n)
    {
        throw std::runtime_error("phase kernel/weight sizes are inconsistent");
    }
    if (!std::isfinite(tolerance))
    {
        throw std::runtime_error("phase balance controls are invalid");
    }

    for (label rowI = 0; rowI < n; ++rowI)
    {
        if (rawSymmetricKernel[rowI].size() < n)
        {
            throw std::runtime_error("phase kernel is not square");
        }
        if (!std::isfinite(weights[rowI]) || weights[rowI] == scalar(0))
        {
            throw std::runtime_error("phase quadrature weight is zero or nonfinite");
        }

    }

    scaling.setSize(n);
    scaling = scalar(1);
    newScaling.setSize(n);
    probabilities.setSize(n);
    forAll(probabilities, rowI)
    {
        probabilities[rowI].setSize(n);
    }

    PhaseBalanceMetrics metrics{scalar(0), scalar(0), scalar(0), 0};
    for (label iteration = 1; iteration <= max(maxIterations, label(1)); ++iteration)
    {
        for (label rowI = 0; rowI < n; ++rowI)
        {
            scalar rowSum = 0;
            for (label colI = 0; colI < n; ++colI)
            {
                rowSum +=
                    weights[colI]*scaling[rowI]
                   *rawSymmetricKernel[rowI][colI]*scaling[colI];
            }
            if (!std::isfinite(rowSum) || rowSum <= scalar(0))
            {
                throw std::runtime_error("phase kernel row sum has an undefined square root or zero divisor");
            }
            newScaling[rowI] = scaling[rowI]/std::sqrt(rowSum);
            if (!std::isfinite(newScaling[rowI]))
            {
                throw std::runtime_error("phase diagonal scaling is zero or nonfinite");
            }
        }
        scaling = newScaling;
        scalar maxScalingCorrection = scalar(0);
        for (label rowI = 0; rowI < n; ++rowI)
        {
            maxScalingCorrection = max
            (
                maxScalingCorrection,
                mag(scaling[rowI] - scalar(1))
            );
            for (label colI = 0; colI < n; ++colI)
            {
                probabilities[rowI][colI] =
                    weights[colI]*scaling[rowI]
                   *rawSymmetricKernel[rowI][colI]*scaling[colI];
            }
        }

        metrics = measurePhaseBalance(probabilities, weights);
        metrics.iterations = iteration;
        metrics.maxRelativeScalingCorrection = maxScalingCorrection;

        if
        (
            metrics.maxRowError <= tolerance
         && metrics.maxWeightedColumnRelativeError <= tolerance
        )
        {
            return metrics;
        }
    }

    return metrics;
}

inline PhaseBalanceResult balanceSymmetricPhaseKernel
(
    const List<scalarField>& rawSymmetricKernel,
    const scalarField& weights,
    const scalar tolerance,
    const label maxIterations
)
{
    scalarField scaling;
    scalarField newScaling;
    List<scalarField> probabilities;
    const PhaseBalanceMetrics metrics = balanceSymmetricPhaseKernelInPlace
    (
        rawSymmetricKernel,
        weights,
        tolerance,
        maxIterations,
        scaling,
        newScaling,
        probabilities
    );
    return PhaseBalanceResult{probabilities, metrics};
}

}                        
}                  

#endif
