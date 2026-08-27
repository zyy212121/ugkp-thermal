#include "../RiemannBoundaryState.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

using ugkpboundary::Primitive;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool close(const double actual, const double expected, const double tolerance = 2.0e-12)
{
    return std::fabs(actual - expected)
        <= tolerance
          *std::fmax(1.0, std::fmax(std::fabs(actual), std::fabs(expected)));
}

Primitive isentropicState
(
    const double mach,
    const double totalPressure,
    const double totalTemperature,
    const double gamma,
    const double gasConstant
)
{
    const double factor = 1.0 + 0.5*(gamma - 1.0)*mach*mach;
    const double temperature = totalTemperature/factor;
    const double pressure =
        totalPressure/std::pow(factor, gamma/(gamma - 1.0));
    const double speed = mach*std::sqrt(gamma*gasConstant*temperature);
    return Primitive
    {
        pressure/(gasConstant*temperature),
        speed,
        0.0,
        0.0,
        pressure,
        temperature
    };
}

void testRecoversSubsonicTotalConditionState()
{
    const double gamma = 1.2;
    const double gasConstant = 287.0;
    const double totalPressure = 1.1e6;
    const double totalTemperature = 3600.0;
    const Primitive expected = isentropicState
    (
        0.1, totalPressure, totalTemperature, gamma, gasConstant
    );
    const Primitive actual = ugkpboundary::totalConditionInletState
    (
        expected,
        -1.0,
        0.0,
        0.0,
        totalPressure,
        totalTemperature,
        gamma,
        gasConstant,
        1.0e-12,
        1.0
    );
    require(close(actual.ux, expected.ux), "subsonic inlet velocity");
    require(close(actual.p, expected.p), "subsonic static pressure");
    require(close(actual.T, expected.T), "subsonic static temperature");
    require(close(actual.rho, expected.rho), "subsonic density");
}

void testColdStartChokesWithoutNonphysicalState()
{
    const Primitive coldOwner
    {
        101325.0/(287.0*300.0), 0.0, 0.0, 0.0, 101325.0, 300.0
    };
    const Primitive inlet = ugkpboundary::totalConditionInletState
    (
        coldOwner,
        -1.0,
        0.0,
        0.0,
        252063.4,
        3600.0,
        1.2,
        287.0,
        1.0e-12,
        1.0
    );
    const double mach =
        std::fabs(inlet.ux)/std::sqrt(1.2*287.0*inlet.T);
    require(close(mach, 1.0), "cold start must use the choked limit");
    require(inlet.p > 0.0 && inlet.T > 0.0 && inlet.rho > 0.0, "positive state");
    require(close(inlet.p, inlet.rho*287.0*inlet.T), "ideal-gas closure");
}

void testHighInvariantProducesStagnantReservoirFace()
{
    const Primitive owner
    {
        1.0e6/(287.0*3600.0), -200.0, 0.0, 0.0, 1.0e6, 3600.0
    };
    const Primitive inlet = ugkpboundary::totalConditionInletState
    (
        owner,
        -1.0,
        0.0,
        0.0,
        1.1e6,
        3600.0,
        1.2,
        287.0,
        1.0e-12,
        1.0
    );
    require(inlet.ux >= 0.0, "inlet direction must point into the domain");
    require(inlet.p <= 1.1e6, "static pressure cannot exceed total pressure");
    require(inlet.T <= 3600.0, "static temperature cannot exceed total temperature");
}

}             

__global__ void compileDeviceTotalConditionInlet(Primitive* value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *value = ugkpboundary::totalConditionInletState
        (
            Primitive{1.0, 0.0, 0.0, 0.0, 1.0e5, 300.0},
            -1.0,
            0.0,
            0.0,
            2.0e5,
            600.0,
            1.4,
            287.0,
            1.0e-12,
            1.0
        );
    }
}

int main()
{
    testRecoversSubsonicTotalConditionState();
    testColdStartChokesWithoutNonphysicalState();
    testHighInvariantProducesStagnantReservoirFace();
    std::cout << "PASS: Riemann total-condition inlet host/device tests\n";
    return 0;
}
