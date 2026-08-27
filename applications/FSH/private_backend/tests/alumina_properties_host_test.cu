#include "../../thermal/GpuAluminaProperties.H"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace
{
bool closeRelative(const double a, const double b, const double tolerance)
{
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= tolerance*scale;
}
}

int main()
{
    using namespace Foam::gpuThermal;

    const auto cold = liquidAluminaProperties(2327.0);
    const auto middle = liquidAluminaProperties(3000.0);
    const auto hot = liquidAluminaProperties(4000.0);
    assert(cold.valid && middle.valid && hot.valid);
    assert(cold.densityKgM3 > middle.densityKgM3);
    assert(middle.densityKgM3 > hot.densityKgM3);
    assert(cold.surfaceTensionNM > hot.surfaceTensionNM);
    assert(cold.viscosityPaS > hot.viscosityPaS);
    assert(cold.specificHeatJkgK < hot.specificHeatJkgK);
    assert(closeRelative(cold.thermalConductivityWmK, 2.70, 1.0e-14));
    assert(closeRelative(hot.thermalConductivityWmK, 2.70, 1.0e-14));

    const auto belowRange = liquidAluminaProperties(750.0);
    const auto aboveRange = liquidAluminaProperties(5000.0);
    assert(belowRange.valid && aboveRange.valid);
    assert(closeRelative(belowRange.densityKgM3, cold.densityKgM3, 1.0e-14));
    assert(closeRelative(aboveRange.densityKgM3, hot.densityKgM3, 1.0e-14));

    const double h3500 = aluminaSpecificEnthalpyJkg(3500.0);
    const double h3000 = aluminaSpecificEnthalpyJkg(3000.0);
    assert(h3500 > h3000);
    const double meanCp = (h3500 - h3000)/500.0;
    assert(meanCp > middle.specificHeatJkgK);
    assert(meanCp < hot.specificHeatJkgK);
    assert(aluminaSpecificEnthalpyJkg(-1.0) < 0.0);
    for (const double temperature : {750.0, 2327.0, 3000.0, 4000.0, 5000.0})
    {
        const double enthalpy = aluminaSpecificEnthalpyJkg(temperature);
        const double recovered =
            aluminaTemperatureFromSpecificEnthalpyK(enthalpy);
        assert(closeRelative(recovered, temperature, 1.0e-12));
    }
    assert(aluminaTemperatureFromSpecificEnthalpyK(-1.0) < 0.0);

    std::cout << "Alumina properties host test: PASS\n";
    return 0;
}
