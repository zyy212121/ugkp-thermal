#include "../../thermal/GpuParticleWallContactHeat.H"

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

    const double diameter = 50.0e-6;
    const double parcelMass = 5.0e-9;
    const double depositionArea = particleWallContactPi*diameter*diameter/4.0;
    const double representedArea = representedDepositionContactArea(
        2800.0, diameter, depositionArea, parcelMass
    );
    const double physicalMass = (particleWallContactPi/6.0)*2800.0
        *diameter*diameter*diameter;
    assert(representedArea > 0.0);
    assert(closeRelative(
        representedArea,
        depositionArea*parcelMass/physicalMass,
        1.0e-12
    ));

    const auto contact = lumpedParticleWallInterfaceContact(
        2500.0, 600.0, physicalMass, parcelMass, 1.0e-4
    );
    assert(contact.valid);
    assert(contact.particleTemperatureK < 2500.0);
    assert(contact.wallEnergyJ > 0.0);

    std::cout << "Particle wall contact heat host test: PASS\n";
    return 0;
}

