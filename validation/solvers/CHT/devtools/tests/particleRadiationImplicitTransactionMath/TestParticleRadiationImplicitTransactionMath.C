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


    const auto raw=conservativeBoundaryEnergyScale(-60,100,160,1e-12);
    require(raw.scale==1 && raw.residualJ==40,"raw exchange must not be balanced");
    const auto zero=conservativeBoundaryEnergyScale(-1,0,1,1e-12);
    require(zero.scale==1 && zero.residualJ==-1,"zero boundary must preserve actual imbalance");
    const auto nonfinite=conservativeBoundaryEnergyScale
        (std::numeric_limits<scalar>::quiet_NaN(),1,1,1e-12);
    require(nonfinite.scale==1 && std::isnan(nonfinite.residualJ),"nonfinite balance must remain visible");

    std::cout << "PASS: raw-energy implicit particle-radiation transaction math\n";
    return 0;
}
