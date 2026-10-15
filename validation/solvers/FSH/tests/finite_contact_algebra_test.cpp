#include "GpuFiniteWallContact.H"
#include "GpuParticleWallContactHeat.H"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    using namespace Foam::gpuThermal;
    const double cosine145 = std::cos(145.0*finiteContactPi/180.0);

                                                                        
                                                                              
                                                                        
    const FiniteWallContactImpact captured = evaluateFiniteWallContactImpact
    (
        3505.4135874549224,
        2.1552432909104658e-5,
        3.1501331293961798,
        cosine145
    );
    assert(captured.valid);
    assert(captured.betaMax >= 1.0);
    assert(captured.maximumAreaM2 > 0.0);
    assert(captured.contactDurationS > 0.0);

                                                                        
    const FiniteWallContactImpact regular = evaluateFiniteWallContactImpact
    (
        3500.0,
        120.0e-6,
        20.0,
        cosine145
    );
    assert(regular.valid);
    assert(regular.betaMax > 1.0);
    assert(std::isfinite(regular.maximumAreaM2));
    assert(std::isfinite(regular.contactDurationS));

                                                                          
                                                                              
                                                                            
    const double defaultEffusivity = combinedContactEffusivity(3500.0);
    const double explicitEffusivity = combinedContactEffusivity
    (
        3500.0,
        1800.0,
        710.0,
        100.0
    );
    const double lowerConductivityEffusivity = combinedContactEffusivity
    (
        3500.0,
        1800.0,
        710.0,
        50.0
    );
    assert(defaultEffusivity == explicitEffusivity);
    assert(lowerConductivityEffusivity > 0.0);
    assert(lowerConductivityEffusivity < explicitEffusivity);

                                                                             
                                                              
    const FiniteWallContactImpact lowSpeed2 = evaluateFiniteWallContactImpact
    (
        3500.0,
        120.0e-6,
        2.0,
        cosine145
    );
    const FiniteWallContactImpact lowSpeed3 = evaluateFiniteWallContactImpact
    (
        3500.0,
        120.0e-6,
        3.0,
        cosine145
    );
    assert(lowSpeed2.valid);
    assert(lowSpeed3.valid);
    assert(lowSpeed2.betaMax > 1.08 && lowSpeed2.betaMax < 1.14);
    assert(lowSpeed3.betaMax > 1.27 && lowSpeed3.betaMax < 1.34);

    const double cosine30 = std::cos(30.0*finiteContactPi/180.0);
    const double cosine90 = std::cos(90.0*finiteContactPi/180.0);
    const FiniteWallContactImpact angle30 = evaluateFiniteWallContactImpact
    (
        3500.0, 120.0e-6, 2.0, cosine30
    );
    const FiniteWallContactImpact angle90 = evaluateFiniteWallContactImpact
    (
        3500.0, 120.0e-6, 2.0, cosine90
    );
    const FiniteWallContactImpact angle145 = evaluateFiniteWallContactImpact
    (
        3500.0, 120.0e-6, 2.0, cosine145
    );
    assert(angle30.valid && angle90.valid && angle145.valid);
    assert(angle30.betaMax > angle90.betaMax);
    assert(angle90.betaMax > angle145.betaMax);

    const double uncorrected30 =
        0.61*std::pow(23.3 + angle30.weber/angle30.ohnesorge, 1.0/6.0);
    const double oldUpperResidual30 = correctedSpreadResidual
    (
        uncorrected30,
        angle30.weber,
        angle30.ohnesorge,
        1.0 - cosine30
    );
    assert(oldUpperResidual30 <= 0.0);
    assert(angle30.betaMax > uncorrected30);

    const AluminaLiquidProperties lowSpeedMaterial =
        liquidAluminaProperties(3500.0);
    const double expectedSpreading =
        0.5*lowSpeed3.betaMax*120.0e-6/3.0;
    const double expectedCapillaryTime = std::sqrt
    (
        lowSpeedMaterial.densityKgM3*std::pow(120.0e-6, 3)
       /lowSpeedMaterial.surfaceTensionNM
    );
    const double expectedNaturalFrequency = 8.0/expectedCapillaryTime;
    const double expectedDampingRate =
        20.0*lowSpeedMaterial.viscosityPaS
       /(lowSpeedMaterial.densityKgM3*std::pow(120.0e-6, 2));
    const double expectedDampedFrequency = std::sqrt
    (
        expectedNaturalFrequency*expectedNaturalFrequency
      - expectedDampingRate*expectedDampingRate
    );
    const double expectedRecoil =
        (finiteContactPi - std::atan2
        (
            expectedDampedFrequency, expectedDampingRate
        ))/expectedDampedFrequency;
    const double expectedDuration = expectedSpreading + expectedRecoil;
    assert
    (
        std::fabs(lowSpeed3.contactDurationS - expectedDuration)
      <= 1.0e-12*expectedDuration
    );
    assert
    (
        std::fabs
        (
            lowSpeed3.peakTimeFraction
          - expectedSpreading/expectedDuration
        ) <= 1.0e-12
    );

    assert(std::fabs(normalizedKinematicArea(0.2, 0.2) - 1.0) < 1.0e-14);
    assert(std::fabs(normalizedKinematicArea(0.1, 0.2) - 0.75) < 1.0e-14);
    assert(std::fabs(normalizedKinematicArea(0.6, 0.2) - 0.5) < 1.0e-14);

    const double wallEffusivity = std::sqrt(1800.0*710.0*100.0);
    const double age0 = 0.0;
    const double ageMid = 2.5e-6;
    const double age1 = 5.0e-6;
    const double resistance = 2.0e-7;
    const double integralFull = wallInterfaceResistanceTimeIntegral
    (
        age0, age1, resistance, wallEffusivity, true
    );
    const double integralSplit =
        wallInterfaceResistanceTimeIntegral
        (
            age0, ageMid, resistance, wallEffusivity, true
        )
      + wallInterfaceResistanceTimeIntegral
        (
            ageMid, age1, resistance, wallEffusivity, true
        );
    assert(integralFull > 0.0);
    assert(std::fabs(integralFull - integralSplit) <= 1.0e-12*integralFull);
    const double zeroResistanceIntegral = wallInterfaceResistanceTimeIntegral
    (
        age0, age1, 0.0, wallEffusivity, true
    );
    const double zeroResistanceExpected =
        2.0*wallEffusivity*std::sqrt(age1)/std::sqrt(finiteContactPi);
    assert
    (
        std::fabs(zeroResistanceIntegral - zeroResistanceExpected)
      <= 1.0e-12*zeroResistanceExpected
    );

    const double physicalMass = 2.0e-9;
    const double parcelMass = 8.0e-8;
    const double conductanceIntegral = 8.0e-7;
    const ParticleWallContactResult lumped = lumpedParticleWallInterfaceContact
    (
        3500.0,
        750.0,
        physicalMass,
        parcelMass,
        conductanceIntegral
    );
    assert(lumped.valid);
    assert(lumped.particleTemperatureK > 750.0);
    assert(lumped.particleTemperatureK < 3500.0);
    const double parcelEnergyFromEnthalpy =
        parcelMass
       *(
            aluminaSpecificEnthalpyJkg(3500.0)
          - aluminaSpecificEnthalpyJkg(lumped.particleTemperatureK)
        );
    assert
    (
        std::fabs(lumped.wallEnergyJ - parcelEnergyFromEnthalpy)
      <= 1.0e-12*parcelEnergyFromEnthalpy
    );
    std::cout
        << "contact-angle betaMax: 30=" << angle30.betaMax
        << " 90=" << angle90.betaMax
        << " 145=" << angle145.betaMax
        << " oldUpperResidual30=" << oldUpperResidual30
        << '\n';
    return 0;
}
