#include "../../thermal/GpuCapillaryDetachment.H"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace
{

bool closeRelative(const double actual, const double expected, const double tolerance)
{
    return std::abs(actual - expected)
        <= tolerance*std::max(std::abs(expected), 1.0e-30);
}

}

int main()
{
    using Foam::gpuThermal::applyCapillaryContactDamage;
    using Foam::gpuThermal::evaluateCapillaryDetachmentState;
    using Foam::gpuThermal::liquidAluminaProperties;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double cosine145 = std::cos(145.0*pi/180.0);
    const double cosine120 = std::cos(120.0*pi/180.0);
    const double cosine30 = std::cos(30.0*pi/180.0);

    const auto at3500 = liquidAluminaProperties(3500.0);
    assert(at3500.valid);
    assert(closeRelative(at3500.densityKgM3, 2772.9556, 1.0e-12));
    assert(closeRelative(at3500.surfaceTensionNM, 0.6049037, 1.0e-12));
    assert(closeRelative(at3500.viscosityPaS, 0.009335284686802578, 1.0e-12));

    const auto belowRange = liquidAluminaProperties(1000.0);
    const auto atLowerLimit = liquidAluminaProperties(2327.0);
    assert(belowRange.valid && atLowerLimit.valid);
    assert(belowRange.temperatureK == atLowerLimit.temperatureK);
    assert(belowRange.densityKgM3 == atLowerLimit.densityKgM3);

    const auto state =
        evaluateCapillaryDetachmentState
        (
            3500.0, 120.0e-6, 1.0, cosine145
        );
    assert(state.valid);
    assert(state.contactAngleCosine < 0.0);
    assert(state.adhesionSpecificEnergyJkg > 0.0);

    const auto scaled =
        evaluateCapillaryDetachmentState
        (
            3500.0, 120.0e-6, 2.0, cosine145
        );
    assert(scaled.valid);
    assert(closeRelative
    (
        scaled.adhesionSpecificEnergyJkg,
        2.0*state.adhesionSpecificEnergyJkg,
        1.0e-12
    ));
    assert
    (
        !evaluateCapillaryDetachmentState
        (
            3500.0, -1.0, 1.0, cosine145
        ).valid
    );

    const auto state120 = evaluateCapillaryDetachmentState
    (
        3500.0, 120.0e-6, 1.0, cosine120
    );
    const auto state30 = evaluateCapillaryDetachmentState
    (
        3500.0, 120.0e-6, 1.0, cosine30
    );
    assert(state120.valid && state30.valid);
    assert(state30.equilibriumContactAreaM2 > state120.equilibriumContactAreaM2);
    assert(state120.equilibriumContactAreaM2 > state.equilibriumContactAreaM2);
    assert(state30.adhesionSpecificEnergyJkg > state120.adhesionSpecificEnergyJkg);
    assert(state120.adhesionSpecificEnergyJkg > state.adhesionSpecificEnergyJkg);

    const double initialArea = 0.16*state.equilibriumContactAreaM2;
    const double initialBarrier =
        0.16*state.adhesionSpecificEnergyJkg;
    const auto unchanged =
        applyCapillaryContactDamage(state, initialArea, 1.0, 0.0);
    assert(unchanged.valid && !unchanged.detached);
    assert(closeRelative(unchanged.remainingContactAreaM2, initialArea, 1.0e-12));
    assert(unchanged.residualSpecificEnergyJkg == 0.0);

    const auto partial =
        applyCapillaryContactDamage
        (
            state, initialArea, 0.5, 0.125*initialBarrier
        );
    assert(partial.valid && !partial.detached);
    assert(closeRelative
    (
        partial.remainingContactAreaM2,
        0.75*initialArea,
        1.0e-12
    ));
    assert(partial.residualSpecificEnergyJkg == 0.0);

    const auto exact =
        applyCapillaryContactDamage(state, initialArea, 0.5, 0.5*initialBarrier);
    assert(exact.valid && exact.detached);
    assert(exact.remainingContactAreaM2 == 0.0);
    assert
    (
        std::abs(exact.residualSpecificEnergyJkg)
     <= 1.0e-12*initialBarrier
    );

    const auto excess =
        applyCapillaryContactDamage
        (
            state, initialArea, 0.5, 0.7*initialBarrier
        );
    assert(excess.valid && excess.detached);
    assert(excess.remainingContactAreaM2 == 0.0);
    assert(closeRelative
    (
        excess.residualSpecificEnergyJkg,
        0.2*initialBarrier,
        1.0e-12
    ));
    assert(!applyCapillaryContactDamage(state, -1.0, 1.0, 1.0).valid);
    assert(!applyCapillaryContactDamage(state, initialArea, 0.0, 1.0).valid);
    assert(!applyCapillaryContactDamage(state, initialArea, 1.0, -1.0).valid);

    std::cout
        << "Capillary detachment host test: PASS"
        << " angle30Area=" << state30.equilibriumContactAreaM2
        << " angle120Area=" << state120.equilibriumContactAreaM2
        << " angle145Area=" << state.equilibriumContactAreaM2
        << " angle30Adhesion=" << state30.adhesionSpecificEnergyJkg
        << " angle120Adhesion=" << state120.adhesionSpecificEnergyJkg
        << " angle145Adhesion=" << state.adhesionSpecificEnergyJkg
        << '\n';
    return 0;
}
