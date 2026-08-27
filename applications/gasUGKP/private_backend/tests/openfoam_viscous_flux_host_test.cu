#include "../OpenFoamViscousFlux.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

using ugkptransport::Vector3;

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

void testOrthogonalCompactDifference()
{
    const Vector3 ownerCentre{0.0, 0.0, 0.0};
    const Vector3 neighbourCentre{2.0, 0.0, 0.0};
    const Vector3 unitNormal{1.0, 0.0, 0.0};
    const auto geometry = ugkptransport::makeInternalSnGradGeometry
    (
        ownerCentre,
        neighbourCentre,
        unitNormal
    );
    require(close(geometry.delta, 0.5), "orthogonal nonOrthDeltaCoeffs");
    require(close(geometry.correction.x, 0.0), "orthogonal correction x");
    require(close(geometry.correction.y, 0.0), "orthogonal correction y");
    require(close(geometry.correction.z, 0.0), "orthogonal correction z");

    const double snGrad = ugkptransport::correctedSnGrad
    (
        1.0,
        5.0,
        Vector3{20.0, -7.0, 3.0},
        Vector3{-11.0, 6.0, 2.0},
        0.31,
        geometry
    );
    require(close(snGrad, 2.0), "orthogonal compact snGrad");
}

void testSkewedLinearFieldIsExact()
{
    const Vector3 ownerCentre{0.0, 0.0, 0.0};
    const Vector3 neighbourCentre{2.0, 1.0, 0.0};
    const Vector3 unitNormal{1.0, 0.0, 0.0};
    const Vector3 exactGradient{3.0, 4.0, 0.0};
    const auto geometry = ugkptransport::makeInternalSnGradGeometry
    (
        ownerCentre,
        neighbourCentre,
        unitNormal
    );
    const double ownerValue = 7.0;
    const double neighbourValue = 17.0;
    const double snGrad = ugkptransport::correctedSnGrad
    (
        ownerValue,
        neighbourValue,
        exactGradient,
        exactGradient,
        0.73,
        geometry
    );
    require(close(snGrad, 3.0), "skewed affine snGrad");
}

void testOpenFoamSplitNewtonianTraction()
{
    const Vector3 unitNormal{0.6, 0.8, 0.0};
    const Vector3 compactSnGradU{2.0, -3.0, 4.0};
    const Vector3 gradUx{1.0, 2.0, 3.0};
    const Vector3 gradUy{4.0, 5.0, 6.0};
    const Vector3 gradUz{7.0, 8.0, 9.0};
    const Vector3 traction = ugkptransport::openFoamNewtonianTraction
    (
        0.25,
        unitNormal,
        compactSnGradU,
        gradUx,
        gradUy,
        gradUz
    );

                                                             
                                             
                                                                              
                                                                              
    const double divU = 1.0 + 5.0 + 9.0;
    const double expectedX =
        0.25*(2.0 + (0.6*1.0 + 0.8*4.0) - (2.0/3.0)*divU*0.6);
    const double expectedY =
        0.25*(-3.0 + (0.6*2.0 + 0.8*5.0) - (2.0/3.0)*divU*0.8);
    const double expectedZ =
        0.25*(4.0 + (0.6*3.0 + 0.8*6.0));
    require(close(traction.x, expectedX), "OpenFOAM split traction x");
    require(close(traction.y, expectedY), "OpenFOAM split traction y");
    require(close(traction.z, expectedZ), "OpenFOAM split traction z");
}

}             

__global__ void compileDeviceOpenFoamViscousFlux(double* value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        const Vector3 ownerCentre{0.0, 0.0, 0.0};
        const Vector3 neighbourCentre{1.0, 0.0, 0.0};
        const Vector3 unitNormal{1.0, 0.0, 0.0};
        const auto geometry = ugkptransport::makeInternalSnGradGeometry
        (
            ownerCentre,
            neighbourCentre,
            unitNormal
        );
        *value = ugkptransport::correctedSnGrad
        (
            0.0,
            1.0,
            Vector3{0.0, 0.0, 0.0},
            Vector3{2.0, 0.0, 0.0},
            0.75,
            geometry
        );
    }
}

int main()
{
    testOrthogonalCompactDifference();
    testSkewedLinearFieldIsExact();
    testOpenFoamSplitNewtonianTraction();
    std::cout
        << "PASS: UGKP OpenFOAM viscous/thermal face-gradient tests\n";
    return 0;
}
