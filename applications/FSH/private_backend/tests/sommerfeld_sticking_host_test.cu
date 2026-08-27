#include "../../thermal/GpuSommerfeldSticking.H"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    using Foam::gpuThermal::evaluateSommerfeldImpact;

    const double particleTemperature = 3000.0;
    const double diameter = 50.0e-6;

    const auto zero =
        evaluateSommerfeldImpact
        (
            particleTemperature, 0.0, diameter, 20.0
        );
    assert(zero.valid);
    assert(!zero.deposit);
    assert(zero.weber == 0.0);
    assert(zero.reynolds == 0.0);
    assert(zero.sommerfeld == 0.0);

    const auto low =
        evaluateSommerfeldImpact
        (
            particleTemperature, 1.0, diameter, 20.0
        );
    const auto high =
        evaluateSommerfeldImpact
        (
            particleTemperature, 100.0, diameter, 20.0
        );
    assert(low.valid && !low.deposit);
    assert(high.valid && high.deposit);
    assert(high.sommerfeld > low.sommerfeld);

    const auto thresholdEqual =
        evaluateSommerfeldImpact
        (
            particleTemperature,
            100.0,
            diameter,
            high.sommerfeld
        );
    assert(thresholdEqual.valid);
    assert(thresholdEqual.deposit);

    const auto invalid =
        evaluateSommerfeldImpact(-1.0, 1.0, diameter, 20.0);
    assert(!invalid.valid);

    std::cout << "Sommerfeld sticking host test: PASS\n";
    return 0;
}
