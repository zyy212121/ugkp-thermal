#include "ParticleRadiationImplicitMath.H"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{

using Foam::scalar;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::abort();
    }
}

bool near(const scalar actual, const scalar expected, const scalar tolerance)
{
    return std::abs(actual - expected)
        <= tolerance*std::max({std::abs(actual), std::abs(expected), scalar(1)});
}

template<class Action>
void requireFailure(const Action& action, const char* message)
{
    try
    {
        action();
    }
    catch (const Foam::gpuThermal::ParticleRadiationSolveError&)
    {
        return;
    }
    require(false, message);
}

}             

int main()
{
    using Foam::gpuThermal::detail::conservativeBoundaryEnergyScale;
    using Foam::gpuThermal::detail::solveBoundedImplicitMaterialTemperature;

    const scalar oldTemperatureK = 3600;
    const scalar equilibriumTemperatureK = 1200;
    const scalar capacityJPerK = 300;
    const scalar elapsedTimeS = 0.01;
    const scalar coefficient = 1e-3;
    const auto balance = [=](const scalar temperatureK)
    {
        return coefficient
          *(std::pow(equilibriumTemperatureK, 4) - std::pow(temperatureK, 4));
    };
    const scalar explicitTemperatureK = oldTemperatureK
      + elapsedTimeS*balance(oldTemperatureK)/capacityJPerK;
    require
    (
        explicitTemperatureK < scalar(300),
        "fixture must make explicit cooling cross the legal minimum"
    );
    const auto cooling = solveBoundedImplicitMaterialTemperature
    (
        oldTemperatureK,
        scalar(300),
        scalar(5000),
        capacityJPerK,
        elapsedTimeS,
        200,
        scalar(1e-10),
        scalar(1e-9),
        balance,
        balance
    );
    require
    (
        cooling.temperatureK > equilibriumTemperatureK
     && cooling.temperatureK < oldTemperatureK,
        "implicit cooling must stay between old and equilibrium temperatures"
    );
    require
    (
        near
        (
            capacityJPerK*(cooling.temperatureK-oldTemperatureK),
            elapsedTimeS*balance(cooling.temperatureK),
            scalar(1e-8)
        ),
        "implicit cooling must satisfy the material energy equation"
    );

    const scalar nearEquilibriumOldTemperatureK =
        3581.5427148141689;
    const scalar nearEquilibriumTemperatureK =
        3581.5426983872289;
    const auto nearEquilibriumBalance = [=](const scalar temperatureK)
    {
        return scalar(14.236)
          *(nearEquilibriumTemperatureK - temperatureK);
    };
    const auto nearEquilibrium = solveBoundedImplicitMaterialTemperature
    (
        nearEquilibriumOldTemperatureK,
        scalar(300),
        scalar(5000),
        scalar(0.243),
        scalar(0.01),
        100,
        scalar(1e-10),
        scalar(1e-9),
        nearEquilibriumBalance,
        nearEquilibriumBalance
    );
    require
    (
        nearEquilibrium.temperatureK >= nearEquilibriumTemperatureK
     && nearEquilibrium.temperatureK <= nearEquilibriumOldTemperatureK,
        "near-equilibrium root must converge at floating-point resolution"
    );

    const auto conservative = conservativeBoundaryEnergyScale
    (
        scalar(-60), scalar(100), scalar(160), scalar(1e-12)
    );
    require(near(conservative.scale, scalar(0.6), scalar(1e-14)),
        "boundary scale must balance accepted particle energy");
    require(near(conservative.residualJ, scalar(0), scalar(1e-14)),
        "balanced transaction residual must be zero");

    const auto zero = conservativeBoundaryEnergyScale
    (
        scalar(0), scalar(0), scalar(0), scalar(1e-12)
    );
    require(zero.scale == scalar(1) && zero.residualJ == scalar(0),
        "zero transaction must retain unit scale");

    requireFailure
    (
        []()
        {
            conservativeBoundaryEnergyScale
            (
                scalar(60), scalar(100), scalar(160), scalar(1e-12)
            );
        },
        "negative boundary scale must be rejected"
    );
    requireFailure
    (
        []()
        {
            conservativeBoundaryEnergyScale
            (
                scalar(-120), scalar(100), scalar(220), scalar(1e-12)
            );
        },
        "boundary-energy amplification must be rejected"
    );
    requireFailure
    (
        []()
        {
            conservativeBoundaryEnergyScale
            (
                scalar(-1), scalar(0), scalar(1), scalar(1e-12)
            );
        },
        "nonzero particle energy with zero boundary energy must be rejected"
    );
    requireFailure
    (
        []()
        {
            conservativeBoundaryEnergyScale
            (
                std::numeric_limits<scalar>::quiet_NaN(),
                scalar(1), scalar(1), scalar(1e-12)
            );
        },
        "nonfinite transaction energy must be rejected"
    );

    std::cout << "PASS: conservative implicit particle-radiation transaction math\n";
    return 0;
}
