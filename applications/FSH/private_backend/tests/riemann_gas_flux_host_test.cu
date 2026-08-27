#include "../RiemannGasFlux.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

using ugkpriemann::FluxResult;
using ugkpriemann::DensityGradient;
using ugkpriemann::Primitive;
using ugkpriemann::Scheme;

constexpr double gammaGas = 1.4;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool closeRelative
(
    const double actual,
    const double expected,
    const double tolerance
)
{
    const double scale =
        std::max(1.0, std::max(std::abs(actual), std::abs(expected)));
    return std::abs(actual - expected) <= tolerance*scale;
}

void requireFinite(const FluxResult& result, const char* message)
{
    require(result.valid, message);
    require(std::isfinite(result.maxSignalSpeed), message);
    require(result.maxSignalSpeed >= 0.0, message);
    for (const double component : result.flux)
    {
        require(std::isfinite(component), message);
    }
}

void physicalFluxX(const Primitive& state, double flux[5])
{
    const double rhoEnergy =
        state.p/(gammaGas - 1.0)
      + 0.5*state.rho*
       (
           state.ux*state.ux
         + state.uy*state.uy
         + state.uz*state.uz
       );
    flux[0] = state.rho*state.ux;
    flux[1] = state.rho*state.ux*state.ux + state.p;
    flux[2] = state.rho*state.uy*state.ux;
    flux[3] = state.rho*state.uz*state.ux;
    flux[4] = (rhoEnergy + state.p)*state.ux;
}

void testCreateCodeMapping()
{
    const Scheme expected[]
    {
        Scheme::RusanovTadmor,
        Scheme::HllKurganov,
        Scheme::HLLE,
        Scheme::HLLC,
        Scheme::Roe,
        Scheme::HLLEM,
        Scheme::HLLC_ADC,
        Scheme::SLAU2,
        Scheme::SLAU2_2
    };
    for (int createCode = 1; createCode <= 9; ++createCode)
    {
        Scheme scheme = Scheme::RusanovTadmor;
        require
        (
            ugkpriemann::schemeFromCreateCode(createCode, scheme),
            "Create code 1..9 must map to a gas flux scheme"
        );
        require
        (
            scheme == expected[createCode - 1],
            "Create code must map to the documented zero-based enum"
        );
        require
        (
            ugkpriemann::createCodeFromScheme(scheme) == createCode,
            "Create-code mapping must round trip"
        );
    }
    Scheme fallback = Scheme::Roe;
    require
    (
        !ugkpriemann::schemeFromCreateCode(0, fallback)
     && fallback == Scheme::RusanovTadmor,
        "invalid Create code must be rejected with a robust default"
    );
}

void testEqualState()
{
    const Primitive state{1.17, 230.0, -41.0, 17.0, 1.23e5};
    double expected[5];
    physicalFluxX(state, expected);
    const Scheme schemes[]
    {
        Scheme::RusanovTadmor,
        Scheme::HllKurganov,
        Scheme::HLLE,
        Scheme::HLLC,
        Scheme::Roe,
        Scheme::HLLEM,
        Scheme::HLLC_ADC,
        Scheme::SLAU2
    };
    for (const Scheme scheme : schemes)
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            state, state, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "equal-state flux must be finite");
        require
        (
            !result.usedFallback,
            "equal-state flux must not require a fallback"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative(result.flux[component], expected[component], 2.0e-13),
                "equal-state numerical flux must equal physical Euler flux"
            );
        }
    }
}

void testContactResolution()
{
    const Primitive left{1.0, 80.0, 14.0, -7.0, 1.0e5};
    const Primitive right{0.25, 80.0, 14.0, -7.0, 1.0e5};
    double expected[5];
    physicalFluxX(left, expected);
    for
    (
        const Scheme scheme
        :
        {
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC,
            Scheme::SLAU2
        }
    )
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "contact flux must be finite");
        require
        (
            !result.usedFallback,
            "resolved contact must not trigger fallback"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative(result.flux[component], expected[component], 2.0e-12),
                "low-dissipation schemes must exactly transport an isolated contact"
            );
        }
    }

    for
    (
        const Scheme scheme :
        {
            Scheme::RusanovTadmor,
            Scheme::HllKurganov,
            Scheme::HLLE
        }
    )
    {
        requireFinite
        (
            ugkpriemann::fluxUnitArea
            (
                left, right, 1.0, 0.0, 0.0, gammaGas, scheme
            ),
            "diffusive contact flux must remain finite"
        );
    }
}

void testArbitraryNormalStationaryHllcShearContact()
{
    const double inverseLength = 1.0/std::sqrt(6.0);
    const double nx = inverseLength;
    const double ny = 2.0*inverseLength;
    const double nz = -inverseLength;
                                                                            
                                          
    const Primitive left{1.2, 2.0, -1.0, 0.0, 7.5e4};
    const Primitive right{0.4, -1.0, 1.0, 1.0, 7.5e4};
    const FluxResult result = ugkpriemann::fluxUnitArea
    (
        left, right, nx, ny, nz, gammaGas, Scheme::HLLC
    );
    requireFinite(result, "stationary HLLC shear/contact must be finite");
    require(!result.usedFallback, "resolved HLLC contact must not fall back");
    const double expected[5]
    {
        0.0,
        left.p*nx,
        left.p*ny,
        left.p*nz,
        0.0
    };
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(result.flux[component], expected[component], 2.0e-12),
            "HLLC must preserve a stationary arbitrary-normal shear/contact"
        );
    }
}

void testArbitraryNormalStationaryRoeShearContactWithEntropyFix()
{
    const double inverseLength = 1.0/std::sqrt(6.0);
    const double nx = inverseLength;
    const double ny = 2.0*inverseLength;
    const double nz = -inverseLength;
    const Primitive left{1.2, 2.0, -1.0, 0.0, 7.5e4};
    const Primitive right{0.4, -1.0, 1.0, 1.0, 7.5e4};
    const FluxResult result = ugkpriemann::fluxUnitArea
    (
        left,
        right,
        nx,
        ny,
        nz,
        gammaGas,
        Scheme::Roe,
        0.1
    );
    requireFinite(result, "stationary Roe shear/contact must be finite");
    require(!result.usedFallback, "resolved Roe contact must not fall back");
    const double expected[5]
    {
        0.0,
        left.p*nx,
        left.p*ny,
        left.p*nz,
        0.0
    };
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(result.flux[component], expected[component], 2.0e-12),
            "Roe entropy fix must preserve a stationary shear/contact"
        );
    }
}

void testSodStates()
{
    const Primitive left{1.0, 0.0, 0.0, 0.0, 1.0};
    const Primitive right{0.125, 0.0, 0.0, 0.0, 0.1};
    for
    (
        const Scheme scheme :
        {
            Scheme::RusanovTadmor,
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC,
            Scheme::SLAU2
        }
    )
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "Sod flux must remain finite");
        require
        (
            result.maxSignalSpeed > 1.0,
            "Sod flux must report its acoustic signal speed"
        );
        require
        (
            result.flux[0] > 0.0,
            "Sod mass flux must point from the high-pressure state"
        );
    }
}

void testSupersonicUpwind()
{
    const Primitive left{1.0, 900.0, 20.0, -10.0, 1.0e5};
    const Primitive right{0.3, 850.0, -5.0, 2.0, 2.0e4};
    double expected[5];
    physicalFluxX(left, expected);
    for
    (
        const Scheme scheme :
        {
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC
        }
    )
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "supersonic upwind flux must be finite");
        require
        (
            !result.usedFallback,
            "supersonic upwind flux must not fall back"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative(result.flux[component], expected[component], 2.0e-12),
                "supersonic positive flux must use the left state"
            );
        }
    }
}

void testNegativeSupersonicUpwind()
{
    const Primitive left{0.3, -850.0, -5.0, 2.0, 2.0e4};
    const Primitive right{1.0, -900.0, 20.0, -10.0, 1.0e5};
    double expected[5];
    physicalFluxX(right, expected);
    for
    (
        const Scheme scheme :
        {
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC
        }
    )
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "negative supersonic upwind flux must be finite");
        require
        (
            !result.usedFallback,
            "negative supersonic upwind flux must not fall back"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative(result.flux[component], expected[component], 2.0e-12),
                "supersonic negative flux must use the right state"
            );
        }
    }
}

struct Rotation
{
    double m[3][3];
};

void rotateVector
(
    const Rotation& rotation,
    const double x,
    const double y,
    const double z,
    double& resultX,
    double& resultY,
    double& resultZ
)
{
    resultX =
        rotation.m[0][0]*x
      + rotation.m[0][1]*y
      + rotation.m[0][2]*z;
    resultY =
        rotation.m[1][0]*x
      + rotation.m[1][1]*y
      + rotation.m[1][2]*z;
    resultZ =
        rotation.m[2][0]*x
      + rotation.m[2][1]*y
      + rotation.m[2][2]*z;
}

Primitive rotatedPrimitive
(
    const Rotation& rotation,
    const Primitive& state
)
{
    Primitive result = state;
    rotateVector
    (
        rotation,
        state.ux,
        state.uy,
        state.uz,
        result.ux,
        result.uy,
        result.uz
    );
    return result;
}

void testRotationConsistency()
{
                                                                          
    const double ax = 1.0/std::sqrt(6.0);
    const double ay = 2.0/std::sqrt(6.0);
    const double az = -1.0/std::sqrt(6.0);
    const double angle = 0.63;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double oneMinusC = 1.0 - c;
    const Rotation rotation
    {{
        {
            c + ax*ax*oneMinusC,
            ax*ay*oneMinusC - az*s,
            ax*az*oneMinusC + ay*s
        },
        {
            ay*ax*oneMinusC + az*s,
            c + ay*ay*oneMinusC,
            ay*az*oneMinusC - ax*s
        },
        {
            az*ax*oneMinusC - ay*s,
            az*ay*oneMinusC + ax*s,
            c + az*az*oneMinusC
        }
    }};

    const Primitive left{0.83, 310.0, -55.0, 26.0, 9.5e4};
    const Primitive right{0.47, -80.0, 31.0, -19.0, 4.2e4};
    const double nx = 0.3;
    const double ny = -0.4;
    const double nz = std::sqrt(0.75);
    double rotatedNx;
    double rotatedNy;
    double rotatedNz;
    rotateVector
    (
        rotation, nx, ny, nz, rotatedNx, rotatedNy, rotatedNz
    );
    const Primitive rotatedLeft = rotatedPrimitive(rotation, left);
    const Primitive rotatedRight = rotatedPrimitive(rotation, right);

    for
    (
        const Scheme scheme :
        {
            Scheme::RusanovTadmor,
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC,
            Scheme::SLAU2
        }
    )
    {
        const FluxResult reference = ugkpriemann::fluxUnitArea
        (
            left, right, nx, ny, nz, gammaGas, scheme
        );
        const FluxResult rotated = ugkpriemann::fluxUnitArea
        (
            rotatedLeft,
            rotatedRight,
            rotatedNx,
            rotatedNy,
            rotatedNz,
            gammaGas,
            scheme
        );
        const FluxResult reversed = ugkpriemann::fluxUnitArea
        (
            right, left, -nx, -ny, -nz, gammaGas, scheme
        );
        requireFinite(reference, "reference rotated flux must be finite");
        requireFinite(rotated, "rotated flux must be finite");
        requireFinite(reversed, "reversed flux must be finite");
        require
        (
            reference.evaluatedScheme == rotated.evaluatedScheme,
            "rotation must not alter fallback selection"
        );
        require
        (
            reference.evaluatedScheme == reversed.evaluatedScheme,
            "face reversal must not alter fallback selection"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative
                (
                    reference.flux[component],
                    -reversed.flux[component],
                    3.0e-12
                ),
                "swapping states and reversing the normal must reverse flux"
            );
        }
        require
        (
            closeRelative
            (
                reference.maxSignalSpeed,
                rotated.maxSignalSpeed,
                3.0e-13
            ),
            "signal speed must be rotation invariant"
        );
        require
        (
            closeRelative(reference.flux[0], rotated.flux[0], 3.0e-12),
            "mass flux must be rotation invariant"
        );
        require
        (
            closeRelative(reference.flux[4], rotated.flux[4], 3.0e-12),
            "energy flux must be rotation invariant"
        );

        double expectedMomentumX;
        double expectedMomentumY;
        double expectedMomentumZ;
        rotateVector
        (
            rotation,
            reference.flux[1],
            reference.flux[2],
            reference.flux[3],
            expectedMomentumX,
            expectedMomentumY,
            expectedMomentumZ
        );
        require
        (
            closeRelative(rotated.flux[1], expectedMomentumX, 3.0e-12)
         && closeRelative(rotated.flux[2], expectedMomentumY, 3.0e-12)
         && closeRelative(rotated.flux[3], expectedMomentumZ, 3.0e-12),
            "momentum flux must rotate as a vector"
        );
    }
}

void testAreaVectorScaling()
{
    const Primitive left{1.0, 70.0, -12.0, 5.0, 1.0e5};
    const Primitive right{0.8, 40.0, 8.0, -3.0, 8.0e4};
    const FluxResult unit = ugkpriemann::fluxUnitArea
    (
        left, right, 1.0, 0.0, 0.0, gammaGas, Scheme::HLLE
    );
    const FluxResult area = ugkpriemann::fluxAreaVector
    (
        left, right, 3.5, 0.0, 0.0, gammaGas, Scheme::HLLE
    );
    requireFinite(unit, "unit-area flux must be finite");
    requireFinite(area, "area-vector flux must be finite");
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(area.flux[component], 3.5*unit.flux[component], 2.0e-13),
            "area-vector flux must scale by face area"
        );
    }
}

void testExtremeNormalScaleInvariance()
{
    const Primitive left{1.0, 0.8, -0.15, 0.03, 1.0};
    const Primitive right{0.7, -0.2, 0.08, -0.01, 0.6};
    for
    (
        const Scheme scheme
        :
        {
            Scheme::RusanovTadmor,
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC,
            Scheme::SLAU2
        }
    )
    {
        const FluxResult reference = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(reference, "reference normal flux must be finite");
        for (const double scale : {1.0e-200, 1.0e200})
        {
            const FluxResult scaled = ugkpriemann::fluxUnitArea
            (
                left, right, scale, 0.0, 0.0, gammaGas, scheme
            );
            requireFinite(
                scaled,
                "any finite non-zero unit-normal scale must be accepted"
            );
            for (int component = 0; component < 5; ++component)
            {
                require
                (
                    closeRelative
                    (
                        scaled.flux[component],
                        reference.flux[component],
                        5.0e-13
                    ),
                    "unit-area flux must be invariant to normal scale"
                );
            }
        }
    }
}

void testAreaOverflowRejected()
{
    const Primitive huge{1.0, 1.0, 0.0, 0.0, 1.0e300};
    const FluxResult result = ugkpriemann::fluxAreaVector
    (
        huge,
        huge,
        1.0e10,
        0.0,
        0.0,
        gammaGas,
        Scheme::HllKurganov
    );
    require(
        !result.valid,
        "area scaling that overflows a flux component must be rejected"
    );
}

void testRoeAutomaticEntropyFixNearSonic()
{
    const Primitive left{1.0, 1.1, 0.0, 0.0, 1.0};
    const Primitive right{1.0, 1.5, 0.0, 0.0, 1.0};
    const FluxResult first = ugkpriemann::fluxUnitArea
    (
        left,
        right,
        1.0,
        0.0,
        0.0,
        gammaGas,
        Scheme::Roe
    );
    const FluxResult second = ugkpriemann::fluxUnitArea
    (
        left,
        right,
        1.0,
        0.0,
        0.0,
        gammaGas,
        Scheme::Roe
    );
    requireFinite(first, "automatic near-sonic Roe flux must be finite");
    requireFinite(second, "automatic near-sonic Roe flux must be repeatable");
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(first.flux[component], second.flux[component], 2.0e-13),
            "automatic Roe entropy correction must not depend on a user coefficient"
        );
    }
}

void testStrongRarefactionFallback()
{
    const Primitive left{1.0, -1000.0, 0.0, 0.0, 1.0e3};
    const Primitive right{1.0, 1000.0, 0.0, 0.0, 1.0e3};
    for (const Scheme scheme : {Scheme::HLLC, Scheme::Roe})
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left, right, 1.0, 0.0, 0.0, gammaGas, scheme
        );
        requireFinite(result, "strong-rarefaction fallback must remain finite");
        require
        (
            result.usedFallback,
            "inadmissible HLLC/Roe fan must fall back"
        );
        require
        (
            result.evaluatedScheme == Scheme::HLLE
         || result.evaluatedScheme == Scheme::RusanovTadmor,
            "HLLC/Roe fallback must use HLLE or Rusanov"
        );
    }
}

void testCompleteFallbackChainMetadata()
{
    const Primitive left
    {
        0.947471324260861,
        1.0587264622850907,
        0.0,
        0.0,
        798.0311227429978
    };
    const Primitive right
    {
        0.11697221137974612,
        326.780061387132,
        0.0,
        0.0,
        793.3602904200176
    };
    for (const Scheme scheme : {Scheme::HLLC, Scheme::Roe})
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            left,
            right,
            1.0,
            0.0,
            0.0,
            gammaGas,
            scheme,
            1.0e-14,
            500.0
        );
        requireFinite(result, "complete fallback chain must remain finite");
        require(result.usedFallback, "complete fallback flag must be set");
        require
        (
            result.evaluatedScheme == Scheme::RusanovTadmor,
            "HLLC/Roe must fall through HLLE to Rusanov"
        );
    }
}

void testInvalidInputRejected()
{
    const Primitive invalid{-1.0, 0.0, 0.0, 0.0, 1.0};
    const Primitive valid{1.0, 0.0, 0.0, 0.0, 1.0};
    const FluxResult result = ugkpriemann::fluxUnitArea
    (
        invalid, valid, 1.0, 0.0, 0.0, gammaGas, Scheme::HLLE
    );
    require
    (
        !result.valid,
        "invalid primitive input must be rejected instead of silently clamped"
    );
}

void testFloorEqualityAccepted()
{
    const double rhoFloor = 1.0e-12;
    const double pressureFloor = 287.0e-12;
    const Primitive floorState
    {
        rhoFloor,
        0.0,
        0.0,
        0.0,
        pressureFloor
    };
    double expected[5];
    physicalFluxX(floorState, expected);
    for
    (
        const Scheme scheme
        :
        {
            Scheme::RusanovTadmor,
            Scheme::HllKurganov,
            Scheme::HLLE,
            Scheme::HLLC,
            Scheme::Roe,
            Scheme::HLLEM,
            Scheme::HLLC_ADC,
            Scheme::SLAU2
        }
    )
    {
        const FluxResult result = ugkpriemann::fluxUnitArea
        (
            floorState,
            floorState,
            1.0,
            0.0,
            0.0,
            gammaGas,
            scheme,
            rhoFloor,
            pressureFloor
        );
        requireFinite(
            result,
            "a state exactly at configured floors must remain admissible"
        );
        require
        (
            !result.usedFallback && result.evaluatedScheme == scheme,
            "an equal state at the floors must retain the requested scheme"
        );
        for (int component = 0; component < 5; ++component)
        {
            require
            (
                closeRelative(result.flux[component], expected[component], 2.0e-13),
                "floor-equal state must return the physical Euler flux"
            );
        }
    }
}

void testHllcAdcSelectiveAntidiffusionEndpoints()
{
    const double inverseLength = 1.0/std::sqrt(14.0);
    const double nx = inverseLength;
    const double ny = 2.0*inverseLength;
    const double nz = 3.0*inverseLength;
    const Primitive left{1.1, 420.0, -35.0, 18.0, 2.1e5};
    const Primitive right{0.62, 160.0, 22.0, -11.0, 8.0e4};

    const FluxResult hllc = ugkpriemann::fluxUnitArea
    (
        left, right, nx, ny, nz, gammaGas, Scheme::HLLC
    );
    const FluxResult hlle = ugkpriemann::fluxUnitArea
    (
        left, right, nx, ny, nz, gammaGas, Scheme::HLLE
    );
    const FluxResult adcFull = ugkpriemann::fluxUnitArea
    (
        left,
        right,
        nx,
        ny,
        nz,
        gammaGas,
        Scheme::HLLC_ADC,
        1.0e-14,
        1.0e-12,
        1.0
    );
    const FluxResult adcSuppressed = ugkpriemann::fluxUnitArea
    (
        left,
        right,
        nx,
        ny,
        nz,
        gammaGas,
        Scheme::HLLC_ADC,
        1.0e-14,
        1.0e-12,
        0.0
    );
    requireFinite(hllc, "HLLC comparison flux must be finite");
    requireFinite(hlle, "HLLE comparison flux must be finite");
    requireFinite(adcFull, "full HLLC-ADC flux must be finite");
    requireFinite(adcSuppressed, "suppressed HLLC-ADC flux must be finite");

    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(adcFull.flux[component], hllc.flux[component], 2.0e-13),
            "omega=1 HLLC-ADC must recover HLLC exactly"
        );
    }

    require
    (
        closeRelative(adcSuppressed.flux[0], hlle.flux[0], 2.0e-13),
        "omega=0 must use the HLLE mass flux"
    );
    const double hllcNormalMomentum =
        hllc.flux[1]*nx + hllc.flux[2]*ny + hllc.flux[3]*nz;
    const double hlleNormalMomentum =
        hlle.flux[1]*nx + hlle.flux[2]*ny + hlle.flux[3]*nz;
    const double adcNormalMomentum =
        adcSuppressed.flux[1]*nx
      + adcSuppressed.flux[2]*ny
      + adcSuppressed.flux[3]*nz;
    require
    (
        closeRelative(adcNormalMomentum, hlleNormalMomentum, 2.0e-13),
        "omega=0 must use the HLLE interface-normal momentum flux"
    );
    for (int component = 0; component < 3; ++component)
    {
        const double normal = component == 0 ? nx : (component == 1 ? ny : nz);
        const double hllcTangential =
            hllc.flux[component + 1] - hllcNormalMomentum*normal;
        const double adcTangential =
            adcSuppressed.flux[component + 1] - adcNormalMomentum*normal;
        require
        (
            closeRelative(adcTangential, hllcTangential, 2.0e-13),
            "HLLC-ADC must retain HLLC tangential momentum antidiffusion"
        );
    }
    require
    (
        closeRelative(adcSuppressed.flux[4], hllc.flux[4], 2.0e-13),
        "HLLC-ADC must retain the HLLC energy flux"
    );
}

void testSlau22RequiresDensityGradients()
{
    const Primitive left{1.0, 120.0, 7.0, -3.0, 1.0e5};
    const Primitive right{0.8, 95.0, -2.0, 4.0, 8.0e4};
    const FluxResult unit = ugkpriemann::fluxUnitArea
    (
        left, right, 1.0, 0.0, 0.0, gammaGas, Scheme::SLAU2_2
    );
    const FluxResult area = ugkpriemann::fluxAreaVector
    (
        left, right, 2.0, 0.0, 0.0, gammaGas, Scheme::SLAU2_2
    );
    require(!unit.valid, "generic unit-area SLAU2.2 must reject missing gradients");
    require(!area.valid, "generic area-vector SLAU2.2 must reject missing gradients");
    const DensityGradient leftGradient{0.7, -0.2, 0.1};
    const DensityGradient rightGradient{0.4, 0.3, -0.1};
    requireFinite
    (
        ugkpriemann::slau22FluxUnitNormal
        (
            left,
            right,
            leftGradient,
            rightGradient,
            1.0,
            0.0,
            0.0,
            gammaGas,
            1.0e-14,
            1.0e-12
        ),
        "dedicated SLAU2.2 flux with gradients must be finite"
    );
}

void testSlau22EqualAndContactStates()
{
    const DensityGradient gradient{0.3, -0.4, 0.2};
    const Primitive equal{1.17, 230.0, -41.0, 17.0, 1.23e5};
    double equalExpected[5];
    physicalFluxX(equal, equalExpected);
    const FluxResult equalResult = ugkpriemann::slau22FluxUnitNormal
    (
        equal,
        equal,
        gradient,
        gradient,
        1.0,
        0.0,
        0.0,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    requireFinite(equalResult, "equal-state SLAU2.2 flux must be finite");
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(equalResult.flux[component], equalExpected[component], 2.0e-13),
            "equal-state SLAU2.2 flux must equal the physical flux"
        );
    }

    const Primitive left{1.0, 80.0, 14.0, -7.0, 1.0e5};
    const Primitive right{0.25, 80.0, 14.0, -7.0, 1.0e5};
    double contactExpected[5];
    physicalFluxX(left, contactExpected);
    const FluxResult contact = ugkpriemann::slau22FluxUnitNormal
    (
        left,
        right,
        gradient,
        gradient,
        1.0,
        0.0,
        0.0,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    requireFinite(contact, "contact SLAU2.2 flux must be finite");
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(contact.flux[component], contactExpected[component], 2.0e-12),
            "SLAU2.2 must transport an isolated contact"
        );
    }
}

void testSlau22RotationAndNormalReversal()
{
    const Rotation rotation
    {{
        {0.0, -1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0}
    }};
    const Primitive left{0.83, 310.0, -55.0, 26.0, 9.5e4};
    const Primitive right{0.47, -80.0, 31.0, -19.0, 4.2e4};
    const DensityGradient leftGradient{0.7, -0.2, 0.1};
    const DensityGradient rightGradient{0.4, 0.3, -0.1};
    const double nx = 0.3;
    const double ny = -0.4;
    const double nz = std::sqrt(0.75);
    const FluxResult reference = ugkpriemann::slau22FluxUnitNormal
    (
        left,
        right,
        leftGradient,
        rightGradient,
        nx,
        ny,
        nz,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    const FluxResult reversed = ugkpriemann::slau22FluxUnitNormal
    (
        right,
        left,
        rightGradient,
        leftGradient,
        -nx,
        -ny,
        -nz,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    requireFinite(reference, "reference SLAU2.2 flux must be finite");
    requireFinite(reversed, "normal-reversed SLAU2.2 flux must be finite");
    for (int component = 0; component < 5; ++component)
    {
        require
        (
            closeRelative(reversed.flux[component], -reference.flux[component], 2.0e-12),
            "SLAU2.2 must reverse flux when states and normal are reversed"
        );
    }

    double rotatedNx;
    double rotatedNy;
    double rotatedNz;
    double rotatedGradLX;
    double rotatedGradLY;
    double rotatedGradLZ;
    double rotatedGradRX;
    double rotatedGradRY;
    double rotatedGradRZ;
    rotateVector(rotation, nx, ny, nz, rotatedNx, rotatedNy, rotatedNz);
    rotateVector
    (
        rotation,
        leftGradient.x,
        leftGradient.y,
        leftGradient.z,
        rotatedGradLX,
        rotatedGradLY,
        rotatedGradLZ
    );
    rotateVector
    (
        rotation,
        rightGradient.x,
        rightGradient.y,
        rightGradient.z,
        rotatedGradRX,
        rotatedGradRY,
        rotatedGradRZ
    );
    const FluxResult rotated = ugkpriemann::slau22FluxUnitNormal
    (
        rotatedPrimitive(rotation, left),
        rotatedPrimitive(rotation, right),
        DensityGradient{rotatedGradLX, rotatedGradLY, rotatedGradLZ},
        DensityGradient{rotatedGradRX, rotatedGradRY, rotatedGradRZ},
        rotatedNx,
        rotatedNy,
        rotatedNz,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    requireFinite(rotated, "rotated SLAU2.2 flux must be finite");
    double expectedMomX;
    double expectedMomY;
    double expectedMomZ;
    rotateVector
    (
        rotation,
        reference.flux[1],
        reference.flux[2],
        reference.flux[3],
        expectedMomX,
        expectedMomY,
        expectedMomZ
    );
    require(closeRelative(rotated.flux[0], reference.flux[0], 2.0e-12), "SLAU2.2 mass flux must be rotationally invariant");
    require(closeRelative(rotated.flux[1], expectedMomX, 2.0e-12), "SLAU2.2 x-momentum flux must rotate");
    require(closeRelative(rotated.flux[2], expectedMomY, 2.0e-12), "SLAU2.2 y-momentum flux must rotate");
    require(closeRelative(rotated.flux[3], expectedMomZ, 2.0e-12), "SLAU2.2 z-momentum flux must rotate");
    require(closeRelative(rotated.flux[4], reference.flux[4], 2.0e-12), "SLAU2.2 energy flux must be rotationally invariant");
}

void testSlau22RestoresTransversePressureDamping()
{
                                                                            
                                                                             
                                                                             
    const Primitive left{1.0, 800.0, 0.0, 0.0, 1.0e5};
    const Primitive right{1.1, 800.0, 0.0, 0.0, 1.1e5};
    const double nx = 0.0;
    const double ny = 1.0;
    const double nz = 0.0;
    const DensityGradient radialGradient{0.0, 1.0, 0.0};
    const DensityGradient axialGradient{1.0, 0.0, 0.0};

    const FluxResult baseline = ugkpriemann::fluxUnitArea
    (
        left, right, nx, ny, nz, gammaGas, Scheme::SLAU2
    );
    const FluxResult transverse = ugkpriemann::slau22FluxUnitNormal
    (
        left,
        right,
        radialGradient,
        radialGradient,
        nx,
        ny,
        nz,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    const FluxResult aligned = ugkpriemann::slau22FluxUnitNormal
    (
        left,
        right,
        axialGradient,
        axialGradient,
        nx,
        ny,
        nz,
        gammaGas,
        1.0e-14,
        1.0e-12
    );
    requireFinite(baseline, "baseline transverse SLAU2 flux must be finite");
    requireFinite(transverse, "SLAU2.2 transverse flux must be finite");
    requireFinite(aligned, "SLAU2.2 aligned-gradient flux must be finite");
    require
    (
        closeRelative(baseline.flux[0], 0.0, 2.0e-13),
        "baseline SLAU2 must expose the transverse pressure-decoupling case"
    );
    require
    (
        transverse.flux[0] < 0.0
     && std::abs(transverse.flux[0]) > 1.0,
        "SLAU2.2 must add pressure-jump mass damping on a transverse face"
    );
    require
    (
        closeRelative(aligned.flux[0], baseline.flux[0], 2.0e-13),
        "SLAU2.2 must recover baseline SLAU2 when grad(rho) follows the "
        "supersonic velocity"
    );
}

}             

                                                                            
__global__ void compileDeviceFlux
(
    const Primitive* left,
    const Primitive* right,
    FluxResult* output
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *output = ugkpriemann::fluxUnitArea
        (
            *left,
            *right,
            1.0,
            0.0,
            0.0,
            gammaGas,
            Scheme::HLLC
        );
    }
}

int main()
{
    testCreateCodeMapping();
    testEqualState();
    testContactResolution();
    testArbitraryNormalStationaryHllcShearContact();
    testArbitraryNormalStationaryRoeShearContactWithEntropyFix();
    testSodStates();
    testSupersonicUpwind();
    testNegativeSupersonicUpwind();
    testRotationConsistency();
    testAreaVectorScaling();
    testExtremeNormalScaleInvariance();
    testAreaOverflowRejected();
    testRoeAutomaticEntropyFixNearSonic();
    testStrongRarefactionFallback();
    testCompleteFallbackChainMetadata();
    testInvalidInputRejected();
    testFloorEqualityAccepted();
    testHllcAdcSelectiveAntidiffusionEndpoints();
    testSlau22RequiresDensityGradients();
    testSlau22EqualAndContactStates();
    testSlau22RotationAndNormalReversal();
    testSlau22RestoresTransversePressureDamping();
    std::cout << "PASS: UGKP Riemann gas flux host/device compile tests\n";
    return 0;
}
