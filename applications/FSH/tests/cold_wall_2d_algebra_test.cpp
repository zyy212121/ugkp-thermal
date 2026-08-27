#include "GpuColdWall2DSolidification.H"

#include <cassert>
#include <cmath>

int main()
{
    using namespace Foam::gpuThermal;
    const double diameter = 0.005;
    const double volume = coldWallPi*diameter*diameter*diameter/6.0;
    const double density = 2911.33278;
    const double mass = density*volume;
    const double maximumArea = 4.2507e-4;
    const ColdWall2DGeometry geometry = coldWall2DGeometry
    (
        volume,
        mass,
        maximumArea
    );
    assert(geometry.valid);
    assert(std::fabs(geometry.filmThicknessM*maximumArea - volume) < 1.0e-15);
    assert
    (
        std::fabs
        (
            geometry.ringAreaM2*coldWall2DRadialNodeCount - maximumArea
        ) < 1.0e-15
    );
    assert
    (
        std::fabs
        (
            geometry.nodeMassKg*coldWall2DNodeCount - mass
        ) < 1.0e-15
    );

    const double wetTarget = 0.41*maximumArea;
    double wetSum = 0.0;
    for (int ring = 0; ring < coldWall2DRadialNodeCount; ++ring)
    {
        const double wet = coldWall2DRingWetAreaM2(geometry, wetTarget, ring);
        assert(wet >= 0.0 && wet <= geometry.ringAreaM2);
        wetSum += wet;
    }
    assert(std::fabs(wetSum - wetTarget) < 1.0e-15);
    assert
    (
        coldWall2DRadialFaceConductanceWK(geometry, 3, 5.9, 5.9) > 0.0
    );

    const ColdWallSolidificationParameters parameters =
    {
        2327.0,
        20.0,
        1.16e6,
        3990.0,
        1273.0,
        5.9,
        0.2,
        0.0,
        1,
        4
    };
    double enthalpy[coldWall2DNodeCount];
    double age[coldWall2DRadialNodeCount];
    initialiseColdWall2DProfile(enthalpy, age, 3000.0, parameters);
    const double liquidEnthalpy = coldWallSpecificEnthalpyJkg(3000.0, parameters);
    for (int node = 0; node < coldWall2DNodeCount; ++node)
    {
        assert(std::fabs(enthalpy[node] - liquidEnthalpy) < 1.0e-10);
    }
    for (int ring = 0; ring < coldWall2DRadialNodeCount; ++ring)
    {
        assert(age[ring] == 0.0);
    }

    double ringEnthalpy[coldWall2DAxialNodeCount];
    const double solidEnthalpy = coldWallSpecificEnthalpyJkg(300.0, parameters);
    for (int node = 0; node < coldWall2DAxialNodeCount; ++node)
    {
        ringEnthalpy[node] = node < 4 ? solidEnthalpy : liquidEnthalpy;
    }
    assert
    (
        std::fabs
        (
            coldWall2DConnectedSolidThicknessFraction
            (
                ringEnthalpy,
                parameters
            ) - 0.5
        ) < 1.0e-14
    );
    return 0;
}
