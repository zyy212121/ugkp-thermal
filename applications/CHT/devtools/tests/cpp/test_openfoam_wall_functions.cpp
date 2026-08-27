#include "OpenFoamWallFunctions.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

void requireNear
(
    const char* name,
    const double actual,
    const double expected,
    const double relativeTolerance
)
{
    const double scale = std::max(std::abs(expected), 1.0e-14);
    if (!std::isfinite(actual)
     || std::abs(actual - expected) > relativeTolerance*scale)
    {
        std::cerr
            << name << " expected " << expected
            << " but received " << actual << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}

int main()
{
    {
        const auto result = ugkpwall::spaldingWallState
        (
            654.54,
            4.918204314309113e-05,
            9.29152148664344e-05/4.0906
        );
        requireNear("uTau", result.uTau, 38.83, 0.01);
        requireNear("yPlus", result.yPlus, 84.08, 0.01);
        requireNear("nut", result.nut, 9.06e-05, 0.015);

        const auto transport = ugkpwall::wallSubgridTransport
        (
            4.0906,
            1008.33,
            0.9,
            result.nut
        );
        requireNear
        (
            "muT",
            transport.dynamicViscosity,
            4.0906*result.nut,
            1.0e-12
        );
        requireNear
        (
            "kT",
            transport.thermalConductivity,
            1008.33*transport.dynamicViscosity/0.9,
            1.0e-12
        );
    }

    {
        const auto result = ugkpwall::spaldingWallState
        (
            1.0e-9,
            1.0e-4,
            1.0e-5
        );
        if
        (
            !std::isfinite(result.uTau)
         || !std::isfinite(result.yPlus)
         || !std::isfinite(result.nut)
         || result.uTau < 0.0
         || result.yPlus < 0.0
         || result.nut < 0.0
         || result.nut > 1.0e-12
        )
        {
            std::cerr << "low-shear Spalding limit is invalid\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "OpenFOAM Spalding wall-function algebra: PASS\n";
    return EXIT_SUCCESS;
}
