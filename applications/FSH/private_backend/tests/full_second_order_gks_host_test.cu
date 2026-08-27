#include "../FullSecondOrderGks.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool closeRelative(const double actual, const double expected, const double tol)
{
    const double scale = std::max(1.0, std::max(std::abs(actual), std::abs(expected)));
    return std::abs(actual - expected) <= tol*scale;
}

void zeroGradients(double gradients[3][5])
{
    for (int direction = 0; direction < 3; ++direction)
    {
        for (int component = 0; component < 5; ++component)
        {
            gradients[direction][component] = 0.0;
        }
    }
}

void testHalfMomentPartition()
{
    const ugkpgks::LocalPrimitive state{0.21, 1830.0, -72.0, 19.0, 1.75e5, 0.0};
    const double internalDegrees = 2.0;
    const auto full = ugkpgks::makeMomentCache(state, internalDegrees, 0);
    const auto positive = ugkpgks::makeMomentCache(state, internalDegrees, 1);
    const auto negative = ugkpgks::makeMomentCache(state, internalDegrees, -1);
    for (int order = 0; order <= 6; ++order)
    {
        require
        (
            closeRelative(positive.u[order] + negative.u[order], full.u[order], 2.0e-13),
            "positive and negative half moments must reconstruct the full moment"
        );
    }
}

void testEquilibratedMomentSolve()
{
    const double gasConstant = 287.0;
    const ugkpgks::LocalPrimitive state
    {
        0.012,
        2800.0,
        -400.0,
        90.0,
        0.012*gasConstant*2600.0,
        2600.0
    };
    const auto cache = ugkpgks::makeMomentCache(state, 2.0, 0);
    const double expected[5]{1.0e-4, 3.0e-8, -2.0e-8, 1.0e-8, -4.0e-12};
    double derivative[5];
    ugkpgks::coefficientMoment(cache, expected, 0, 0, 0, derivative);

    double recovered[5];
    require
    (
        ugkpgks::solveMomentCoefficients(cache, derivative, recovered),
        "equilibrated Maxwell moment system must solve in a hot low-density state"
    );

    double reconstructed[5];
    ugkpgks::coefficientMoment(cache, recovered, 0, 0, 0, reconstructed);
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(reconstructed[component], derivative[component], 2.0e-11),
            "reconstructed conservative derivative must match the prescribed derivative"
        );
    }

    double analytic[5];
    require
    (
        ugkpgks::analyticMomentCoefficients(state, 1.4, derivative, analytic),
        "analytic Maxwell coefficient inversion must remain valid"
    );
    ugkpgks::coefficientMoment(cache, analytic, 0, 0, 0, reconstructed);
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(reconstructed[component], derivative[component], 2.0e-11),
            "analytic coefficient inversion must satisfy the Maxwell moment constraints"
        );
    }
}

void testUniformEulerLimit()
{
    constexpr double gamma = 1.4;
    constexpr double gasConstant = 287.0;
    const ugkpgks::LocalPrimitive state
    {
        1.2,
        120.0,
        -15.0,
        8.0,
        101325.0,
        101325.0/(1.2*gasConstant)
    };
    double gradients[3][5];
    zeroGradients(gradients);
    const auto result = ugkpgks::fullSecondOrderFlux
    (
        state,
        state,
        gradients,
        gradients,
        gamma,
        gasConstant,
        1.8e-5,
        0.7,
        2.0e-5,
        1.0e-5
    );
    require(result.valid, "uniform full-GKS state must be valid");

    const double rhoE = state.p/(gamma - 1.0)
      + 0.5*state.rho*(state.u*state.u + state.v*state.v + state.w*state.w);
    const double expected[5]
    {
        state.rho*state.u,
        state.rho*state.u*state.u + state.p,
        state.rho*state.u*state.v,
        state.rho*state.u*state.w,
        (rhoE + state.p)*state.u
    };
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(result.flux[component], expected[component], 2.0e-11),
            "uniform full-GKS flux must reduce to the Euler flux"
        );
    }
}

void testPrandtlCorrectionOnlyChangesEnergy()
{
    constexpr double gamma = 1.4;
    constexpr double gasConstant = 287.0;
    const ugkpgks::LocalPrimitive state
    {
        0.8,
        260.0,
        30.0,
        -10.0,
        85000.0,
        85000.0/(0.8*gasConstant)
    };
    double gradients[3][5];
    zeroGradients(gradients);
    gradients[0][0] = 0.03;
    gradients[0][1] = 8.0;
    gradients[0][2] = 0.7;
    gradients[0][3] = -0.2;
    gradients[0][4] = 2.0e5;
    gradients[1][1] = -1.2;
    gradients[1][2] = 2.0;
    gradients[1][4] = -4.0e4;

    const auto prOne = ugkpgks::fullSecondOrderFlux
    (
        state, state, gradients, gradients,
        gamma, gasConstant, 2.0e-5, 1.0, 1.0e-5, 3.0e-6
    );
    const auto prTarget = ugkpgks::fullSecondOrderFlux
    (
        state, state, gradients, gradients,
        gamma, gasConstant, 2.0e-5, 0.65, 1.0e-5, 3.0e-6
    );
    require(prOne.valid && prTarget.valid, "Prandtl correction test states must be valid");
    for (int component = 0; component < 4; ++component)
    {
        require
        (
            closeRelative(prOne.flux[component], prTarget.flux[component], 2.0e-13),
            "Prandtl correction must not alter mass or momentum flux"
        );
    }
    const double expectedDelta = (1.0/0.65 - 1.0)*prOne.molecularHeatFlux;
    const double actualDelta = prTarget.flux[4] - prOne.flux[4];
    if (!closeRelative(actualDelta, expectedDelta, 1.0e-7))
    {
        std::cerr
            << "Pr correction actual=" << actualDelta
            << " expected=" << expectedDelta
            << " heatFlux=" << prOne.molecularHeatFlux << '\n';
    }
    require
    (
        closeRelative(actualDelta, expectedDelta, 1.0e-7),
        "Prandtl correction must add only the molecular heat-flux correction"
    );
}

void testConstantPressureFourierFluxAcrossPrandtlRange()
{
    constexpr double gamma = 1.4;
    constexpr double gasConstant = 287.0;
    constexpr double temperature = 300.0;
    constexpr double pressure = 100000.0;
    constexpr double viscosity = 0.02;
    constexpr double temperatureGradient = 1.0;
    const double rho = pressure/(gasConstant*temperature);
    const ugkpgks::LocalPrimitive state
    {
        rho, 0.0, 0.0, 0.0, pressure, temperature
    };
    double gradients[3][5];
    zeroGradients(gradients);
                                                                      
                                
    gradients[0][0] = -rho*temperatureGradient/temperature;
    const double cp = gamma*gasConstant/(gamma - 1.0);
    for (const double prandtl : {0.4, 0.6, 0.8, 1.0})
    {
        const auto result = ugkpgks::fullSecondOrderFlux
        (
            state, state, gradients, gradients,
            gamma, gasConstant, viscosity, prandtl, 1.8e-5, 0.0
        );
        require(result.valid, "constant-pressure Fourier state must be valid");
        const double expected = -viscosity*cp*temperatureGradient/prandtl;
        if (!closeRelative(result.flux[4], expected, 5.0e-3))
        {
            std::cerr
                << "Fourier Pr=" << prandtl
                << " actual=" << result.flux[4]
                << " expected=" << expected
                << " molecularHeat=" << result.molecularHeatFlux << '\n';
        }
        require
        (
            closeRelative(result.flux[4], expected, 5.0e-3),
            "complete GKS must recover mu*cp/Pr Fourier conduction"
        );
    }
}

void testStagedMatchesMonolithicFlux()
{
    constexpr double gamma = 1.4;
    constexpr double gasConstant = 287.0;
    const ugkpgks::LocalPrimitive left
    {
        0.36, 740.0, 55.0, -12.0, 1.7e5, 1.7e5/(0.36*gasConstant)
    };
    const ugkpgks::LocalPrimitive right
    {
        0.23, 610.0, -18.0, 24.0, 1.05e5, 1.05e5/(0.23*gasConstant)
    };
    double leftGradients[3][5];
    double rightGradients[3][5];
    zeroGradients(leftGradients);
    zeroGradients(rightGradients);
    const double leftValues[5]{0.07, 38.0, -4.0, 1.5, 8.5e4};
    const double rightValues[5]{-0.03, -12.0, 3.0, -0.8, -4.0e4};
    for (int component = 0; component < 5; ++component)
    {
        leftGradients[0][component] = leftValues[component];
        leftGradients[1][component] = -0.21*leftValues[component];
        leftGradients[2][component] = 0.08*leftValues[component];
        rightGradients[0][component] = rightValues[component];
        rightGradients[1][component] = 0.17*rightValues[component];
        rightGradients[2][component] = -0.11*rightValues[component];
    }

    const auto monolithic = ugkpgks::fullSecondOrderFlux
    (
        left, right, leftGradients, rightGradients,
        gamma, gasConstant, 2.2e-5, 0.72, 8.0e-6, 2.1e-6
    );
    const auto staged = ugkpgks::fullSecondOrderFluxStaged
    (
        left, right, leftGradients, rightGradients,
        gamma, gasConstant, 2.2e-5, 0.72, 8.0e-6, 2.1e-6
    );
    ugkpgks::SideFluxAccumulation productionAccumulation;
    ugkpgks::clearGksSideAccumulation(productionAccumulation);
    require
    (
        ugkpgks::accumulateGksSide
        (
            left, leftGradients, gamma, 1, productionAccumulation
        ),
        "production left-side accumulation must be valid"
    );
    require
    (
        ugkpgks::accumulateGksSide
        (
            right, rightGradients, gamma, -1, productionAccumulation
        ),
        "production right-side accumulation must be valid"
    );
    const auto productionSplit = ugkpgks::finalizeGksFlux
    (
        productionAccumulation,
        gamma, gasConstant, 2.2e-5, 0.72, 8.0e-6, 2.1e-6
    );
    require
    (
        monolithic.valid && staged.valid && productionSplit.valid,
        "all complete GKS arrangements must be valid"
    );
                                                                               
                                                                     
    const double independentReference[5]
    {
        268.40749331888418,
        368527.01527538535,
        13336.924223073607,
        -2514.0644239117819,
        517692575.18574727
    };
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(staged.flux[component], monolithic.flux[component], 3.0e-13),
            "staged and monolithic complete GKS fluxes must match"
        );
        require
        (
            closeRelative
            (
                productionSplit.flux[component],
                monolithic.flux[component],
                3.0e-13
            ),
            "production split accumulation/finalization must match monolithic GKS"
        );
        require
        (
            closeRelative
            (
                staged.molecularViscousFlux[component],
                monolithic.molecularViscousFlux[component],
                3.0e-13
            ),
            "staged and monolithic molecular fluxes must match"
        );
        require
        (
            closeRelative(staged.flux[component], independentReference[component], 8.0e-12),
            "staged CUDA-header flux must match the independent Python reference"
        );
    }
}

}             

int main()
{
    testHalfMomentPartition();
    testEquilibratedMomentSolve();
    testUniformEulerLimit();
    testPrandtlCorrectionOnlyChangesEnergy();
    testConstantPressureFourierFluxAcrossPrandtlRange();
    testStagedMatchesMonolithicFlux();
    std::cout << "UGKP full second-order GKS host tests passed\n";
    return 0;
}
