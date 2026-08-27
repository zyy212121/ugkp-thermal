#include "../../thermal/GpuFiniteWallContact.H"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    using namespace Foam::gpuThermal;
    const double cosine145 = std::cos(145.0*finiteContactPi/180.0);
    const auto impact = evaluateFiniteWallContactImpact
    (
        3500.0, 120.0e-6, 4.0, cosine145
    );
    assert(impact.valid);
    assert(impact.betaMax > 1.0);
    assert(impact.maximumAreaM2 > 0.0);
    assert(impact.contactDurationS > 0.0);
    assert(impact.peakTimeFraction > 0.0 && impact.peakTimeFraction < 1.0);
    assert
    (
        std::fabs
        (
            normalizedKinematicArea
            (
                impact.peakTimeFraction, impact.peakTimeFraction
            ) - 1.0
        ) < 1.0e-12
    );
    assert(normalizedEffectiveArea(impact.peakTimeFraction, impact.peakTimeFraction, 0.25) > 0.74);
    assert(normalizedEffectiveArea(0.01, impact.peakTimeFraction, 0.5) == 0.0);

    std::vector<float> table
    (
        finiteContactTimeTableSize*finiteContactDamageTableSize
       *finiteContactPeakTableSize
    );
    buildFiniteContactRateTable(table.data());
    double integral = 0.0;
    double exactIntegral = 0.0;
    const int steps = 32768;
    for (int i = 0; i < steps; ++i)
    {
        const double theta = (static_cast<double>(i) + 0.5)/steps;
        integral += interpolateFiniteContactRate
        (
            table.data(), theta, impact.peakTimeFraction, 0.0
        )/steps;
        exactIntegral += dimensionlessFiniteContactRate
        (
            theta, impact.peakTimeFraction, 0.0
        )/steps;
    }
    assert(exactIntegral > 0.0);
    assert(std::fabs(integral - exactIntegral)/exactIntegral < 0.02);
    assert(combinedContactEffusivity(3500.0) > 0.0);
    std::cout << "finite wall contact host test: PASS coefficient="
              << integral << '\n';
    return 0;
}
