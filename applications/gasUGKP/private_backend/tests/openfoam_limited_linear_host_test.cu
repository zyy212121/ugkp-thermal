#include "../OpenFoamLimitedLinear.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

using ugkpinterpolation::Vector3;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool close(const double actual, const double expected)
{
    return std::fabs(actual - expected)
        <= 2.0e-13
          *std::fmax(1.0, std::fmax(std::fabs(actual), std::fabs(expected)));
}

void testSmoothLinearFieldUsesCentralWeight()
{
    const double faceValue = ugkpinterpolation::limitedLinearFaceValue
    (
        10.0,
        14.0,
        Vector3{4.0, 0.0, 0.0},
        Vector3{4.0, 0.0, 0.0},
        Vector3{1.0, 0.0, 0.0},
        0.25,
        2.0,
        1.0
    );
    require(close(faceValue, 13.0), "smooth field uses central interpolation");
}

void testPositiveFluxExtremumFallsBackToOwner()
{
    const double faceValue = ugkpinterpolation::limitedLinearFaceValue
    (
        10.0,
        14.0,
        Vector3{0.0, 0.0, 0.0},
        Vector3{4.0, 0.0, 0.0},
        Vector3{1.0, 0.0, 0.0},
        0.25,
        2.0,
        1.0
    );
    require(close(faceValue, 10.0), "positive flux uses owner upwind value");
}

void testNegativeFluxExtremumFallsBackToNeighbour()
{
    const double faceValue = ugkpinterpolation::limitedLinearFaceValue
    (
        10.0,
        14.0,
        Vector3{4.0, 0.0, 0.0},
        Vector3{0.0, 0.0, 0.0},
        Vector3{1.0, 0.0, 0.0},
        0.25,
        -2.0,
        1.0
    );
    require(close(faceValue, 14.0), "negative flux uses neighbour upwind value");
}

void testZeroMassFluxPreservesRiemannEnergyDiffusion()
{
    const double corrected =
        ugkpinterpolation::limitedLinearRiemannEnergyFlux
        (
            -125000.0,
            0.0,
            2.1e6,
            2.4e6
        );
    require
    (
        close(corrected, -125000.0),
        "zero mass flux preserves Riemann energy diffusion"
    );
}

void testUpwindLimitedValuePreservesBaselineFlux()
{
    const double corrected =
        ugkpinterpolation::limitedLinearRiemannEnergyFlux
        (
            3.5e6,
            -2.0,
            1.8e6,
            1.8e6
        );
    require
    (
        close(corrected, 3.5e6),
        "upwind limited value preserves baseline flux"
    );
}

void testSmoothLimitedValueAddsOnlyAdvectiveCorrection()
{
    const double corrected =
        ugkpinterpolation::limitedLinearRiemannEnergyFlux
        (
            50.0,
            2.0,
            10.0,
            13.0
        );
    require
    (
        close(corrected, 56.0),
        "smooth value adds mass-flux advective correction"
    );
}

}             

__global__ void compileDeviceOpenFoamLimitedLinear(double* value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *value = ugkpinterpolation::limitedLinearFaceValue
        (
            10.0,
            14.0,
            Vector3{4.0, 0.0, 0.0},
            Vector3{4.0, 0.0, 0.0},
            Vector3{1.0, 0.0, 0.0},
            0.25,
            2.0,
            1.0
        );
        *value += ugkpinterpolation::limitedLinearRiemannEnergyFlux
        (
            50.0,
            2.0,
            10.0,
            13.0
        );
    }
}

int main()
{
    testSmoothLinearFieldUsesCentralWeight();
    testPositiveFluxExtremumFallsBackToOwner();
    testNegativeFluxExtremumFallsBackToNeighbour();
    testZeroMassFluxPreservesRiemannEnergyDiffusion();
    testUpwindLimitedValuePreservesBaselineFlux();
    testSmoothLimitedValueAddsOnlyAdvectiveCorrection();
    std::cout << "PASS: OpenFOAM limitedLinear host/device tests\n";
    return 0;
}
