#pragma once
#include <cstdint>
namespace ugkp { namespace thermal {
// Physical/scheduling inputs to the existing shared thermal particle owner.
// Ordinary thermal particles use its alumina caloric law; cold-wall models
// retain their existing phase-change profile law. No application identity is
// used to choose numerical operators.
struct ParticlePhaseConfiguration {
    int capacity=0,maxFaceWalkHops=32;
    std::uint64_t randomSeed=1;
    double injectionParcelMass=1e-10;
    double rhoSolid=3990,diameter=1e-4,minimumDiameter=1e-12,maximumDiameter=1;
    double minimumTemperature=1,maximumTemperature=10000;
    int solveTemperature=1,dragModel=1,heatTransferModel=1;
    double dragResidualRe=1e-3,minimumSolidFraction=1e-12,minimumTheta=1e-12;
    int collisionalPressure=0,packingPressure=0,packingIterations=20;
    double collisionalRestitution=.9,pressureKickFraction=.25,packingFraction=.63;
    int csrCellLocalPath=1,heavyReductionMode=0,heavyAutoInterval=100;
    int particleBlockThreads=128,reductionBlockThreads=128,warpAggregatedBinning=0;
};
struct ParticleContactConfiguration {
    // Same public interaction codes as the shared wall library:
    // 1 reboundContact, 3 solidifyingDeposition/coldWall1D, 4 coldWall2D.
    int model=3,heatTransfer=1,wallTransientResistance=1,nonlinearIterations=4;
    double sommerfeldThreshold=20,maximumCoverage=.63;
    double depositionHeatTransferEfficiency=1,reflectionHeatTransferEfficiency=1;
    double adhesionEnergyScale=1,contactAngleDegree=145;
    double meltingTemperature=2327,mushyRange=20,latentHeat=1.16e6;
    double solidDensity=3990,solidSpecificHeat=1273,solidThermalConductivity=5.9;
    double pinningThicknessFraction=.2,interfaceResistance=0;
};
// Allocation and launch choices do not change the physical restart model.
// Keep them in the saved configuration for reproducibility, but permit tuning
// them when restoring the same physical particle state.
inline ParticlePhaseConfiguration particleModelIdentity(ParticlePhaseConfiguration c) {
    const ParticlePhaseConfiguration defaults;
    c.capacity=defaults.capacity;
    c.csrCellLocalPath=defaults.csrCellLocalPath;
    c.heavyReductionMode=defaults.heavyReductionMode;
    c.heavyAutoInterval=defaults.heavyAutoInterval;
    c.particleBlockThreads=defaults.particleBlockThreads;
    c.reductionBlockThreads=defaults.reductionBlockThreads;
    c.warpAggregatedBinning=defaults.warpAggregatedBinning;
    return c;
}
template<class Archive,class Configuration>
void visitParticlePhaseConfiguration(Archive& a,Configuration& c) {
    a(c.capacity);a(c.maxFaceWalkHops);a(c.randomSeed);
    a(c.injectionParcelMass);a(c.rhoSolid);a(c.diameter);a(c.minimumDiameter);a(c.maximumDiameter);
    a(c.minimumTemperature);a(c.maximumTemperature);
    a(c.solveTemperature);a(c.dragModel);a(c.heatTransferModel);a(c.dragResidualRe);
    a(c.minimumSolidFraction);a(c.minimumTheta);a(c.collisionalPressure);a(c.packingPressure);
    a(c.packingIterations);a(c.collisionalRestitution);a(c.pressureKickFraction);a(c.packingFraction);
    a(c.csrCellLocalPath);a(c.heavyReductionMode);a(c.heavyAutoInterval);
    a(c.particleBlockThreads);a(c.reductionBlockThreads);a(c.warpAggregatedBinning);
}
template<class Archive,class Configuration>
void visitParticleContactConfiguration(Archive& a,Configuration& c) {
    a(c.model);a(c.heatTransfer);a(c.wallTransientResistance);a(c.nonlinearIterations);
    a(c.sommerfeldThreshold);a(c.maximumCoverage);a(c.depositionHeatTransferEfficiency);
    a(c.reflectionHeatTransferEfficiency);a(c.adhesionEnergyScale);a(c.contactAngleDegree);
    a(c.meltingTemperature);a(c.mushyRange);a(c.latentHeat);a(c.solidDensity);
    a(c.solidSpecificHeat);a(c.solidThermalConductivity);a(c.pinningThicknessFraction);a(c.interfaceResistance);
}
}}
