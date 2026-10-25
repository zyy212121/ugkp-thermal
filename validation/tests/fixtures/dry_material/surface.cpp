#include "materials/dry/DrySurfaceResponse.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace d = chmt::dry;

void near(double actual, double expected, const char* name, double tolerance=2e-12)
{
    if (!(std::abs(actual-expected) <= tolerance*std::max(1.0, std::abs(expected)))) {
        std::cerr << name << ": " << actual << " != " << expected << '\n';
        std::exit(1);
    }
}

struct GasThermo {
    std::array<double, 3> cp{{1200, 1000, 900}};
    std::array<double, 3> formation{{2e5, -1e6, 0}};
    double enthalpy(std::size_t species, double T) const {
        return formation[species] + cp[species]*T;
    }
};

int main()
{
    std::array<d::CondensedThermo, 1> solid{};
    solid[0].density = 2000;
    solid[0].cp0 = 700;
    solid[0].formationEnergy = 1e5;
    const std::array<double, 1> solidC{{1500}};
    const std::array<double, 3> gasC{{0.2, 0.1, 0.7}};

    // C(s) + O -> CO; gas slots O, CO, inert. Synthetic rates, real mass ratios.
    std::array<d::SurfaceMassAction<1, 3>, 1> reactions{};
    auto& oxidation = reactions[0];
    oxidation.prefactor = 10;
    oxidation.gasOrder[0] = 1;
    oxidation.stoichiometry.condensed[0] = -0.012;
    oxidation.stoichiometry.gas = {{-0.016, 0.028, 0}};
    auto production = d::surfaceProduction(reactions, 1000, solidC, gasC);
    near(production.condensed[0], -0.024, "oxidation solid loss");
    near(production.gas[0], -0.032, "oxidant signed consumption");
    near(production.gas[1], 0.056, "product production");
    near(production.massResidual, 0, "reaction total mass");
    // Independent element accounting, not just total-mass summation.
    near(production.condensed[0] + production.gas[1]*(12.0/28), 0, "carbon balance");
    near(production.gas[0] + production.gas[1]*(16.0/28), 0, "oxygen balance");

    auto activated = oxidation;
    activated.temperaturePower = 0.3;
    activated.activationEnergy = 12000;
    activated.condensedOrder[0] = 0.5;
    const double expectedRate = 10*std::pow(1000, 0.3)
        *std::exp(-12000/(8.31446261815324*1000))*std::sqrt(1500)*0.2;
    near(d::surfaceExtentRate(activated, 1000, solidC, gasC), expectedRate,
        "Arrhenius mass action");
    auto absent = gasC;
    absent[0] = 0;
    near(d::surfaceExtentRate(oxidation, 1000, solidC, absent), 0, "absent reactant");
    activated.prefactor = 0;
    near(d::surfaceExtentRate(activated, 1000, solidC, gasC), 0, "disabled channel");

    d::PoreSurfaceFlux<3> pore{};
    pore.transport[2] = 0.1;
    const auto flux = d::congruentDryInterfaceFlux(production, pore, solid, 0.25);
    near(flux.gasMass, 0.124, "reaction plus pyrolysis blowing");
    near(flux.recessionSpeed, 0.024/(2000*0.75), "porous solid recession");
    near(flux.normalInterfaceSpeed, -flux.recessionSpeed, "recession sign");
    near(flux.gasSpecies[0], -0.032, "no clipping of oxidant uptake");
    near(flux.condensedConsumption[0]-solid[0].density*0.75*flux.recessionSpeed,
        0, "single char front constituent compatibility");

    // Physical-area rates are invariant under an external reference-area
    // representation. The constitutive core must not apply the area ratio twice.
    constexpr double physicalArea = 2, referenceArea = 5;
    auto remapped = production;
    for (double& x : remapped.condensed) x *= physicalArea/referenceArea;
    for (double& x : remapped.gas) x *= physicalArea/referenceArea;
    for (double& x : remapped.condensed) x *= referenceArea/physicalArea;
    for (double& x : remapped.gas) x *= referenceArea/physicalArea;
    near(d::congruentDryInterfaceFlux(remapped, pore, solid, 0.25).recessionSpeed,
        flux.recessionSpeed, "reference area converted before constitutive call");

    // A congruent two-solid front: both constituent supplies independently
    // match their consumed amounts. Selective recession is not this model.
    std::array<d::CondensedThermo, 2> mixedSolid{{solid[0], solid[0]}};
    mixedSolid[1].density = 1000;
    constexpr double prescribedRecession = 2e-5;
    d::SurfaceProduction<2, 3> congruent{};
    congruent.condensed = {{-2000*0.3*prescribedRecession,
                           -1000*0.45*prescribedRecession}};
    congruent.gas[0] = -congruent.condensed[0]-congruent.condensed[1];
    const auto congruentFlux = d::congruentDryInterfaceFlux(congruent, pore, mixedSolid, 0.25);
    near(congruentFlux.recessionSpeed, prescribedRecession, "congruent mixture speed");
    near(congruentFlux.condensedConsumption[0]-2000*0.3*congruentFlux.recessionSpeed,
        0, "first congruent constituent supply");
    near(congruentFlux.condensedConsumption[1]-1000*0.45*congruentFlux.recessionSpeed,
        0, "second congruent constituent supply");

    std::array<d::SurfaceMassAction<1, 3>, 0> none{};
    const auto inert = d::surfaceProduction(none, 1000, solidC, gasC);
    const auto pyrolysis = d::congruentDryInterfaceFlux(inert, pore, solid, 0.25);
    near(pyrolysis.gasMass, 0.1, "pure pyrolysis blowing");
    near(pyrolysis.recessionSpeed, 0, "pyrolysis does not cause surface recession");
    const std::array<d::CondensedThermo, 1> absentSolid{};
    near(d::congruentDryInterfaceFlux(inert, pore, absentSolid, 1).recessionSpeed, 0,
        "no reaction needs no absent-phase density");

    auto deposition = production;
    for (double& x : deposition.condensed) x = -x;
    for (double& x : deposition.gas) x = -x;
    auto growth = d::congruentDryInterfaceFlux(deposition, d::PoreSurfaceFlux<3>{}, solid, 0.25);
    near(growth.recessionSpeed, -flux.recessionSpeed, "deposition gives growth");
    near(growth.gasMass, -0.024, "deposition signed net gas consumption");

    const std::array<double, 3> Y{{0.2, 0.1, 0.7}};
    std::array<double, 3> diffusion{};
    double diffusionSum = 0;
    for (std::size_t s=0; s<3; ++s) {
        diffusion[s] = flux.gasSpecies[s] - flux.gasMass*Y[s];
        diffusionSum += diffusion[s];
    }
    near(diffusionSum, 0, "mass averaged diffusion closure");
    const auto speciesResidual = d::surfaceSpeciesResidual(flux.gasSpecies, Y,
        diffusion, flux.gasMass);
    for (double residual : speciesResidual) near(residual, 0, "GSI species balance");

    GasThermo gas;
    d::DryEnergyInput energy{};
    energy.wallTemperature = 1000;
    energy.gasPressure = 1e5;
    energy.gasSpecificKineticEnergy = 8;
    energy.gasConductiveFlux = 1200;
    energy.gasTractionPower = 10;
    energy.solidConductiveFlux = 300;
    energy.radiationIntoInterface = 50;
    pore.transportEnergy[2] = pore.transport[2]*gas.enthalpy(2, 800);
    const auto balance = d::dryEnergyBalance(flux, pore, solid, gas, energy);
    const double expectedGas = -0.032*gas.enthalpy(0, 1000)
        +0.056*gas.enthalpy(1, 1000)+0.1*gas.enthalpy(2, 1000)
        +0.124*8+1e5*flux.normalInterfaceSpeed+1200-10;
    const double expectedMaterial = 0.024*solid[0].internalEnergy(1000)
        +0.1*gas.enthalpy(2, 800)+300;
    near(balance.gasEnergy, expectedGas, "full outward ALE energy");
    near(balance.materialEnergy, expectedMaterial, "material incoming energy");
    near(balance.residual, expectedGas-expectedMaterial-50, "energy residual");

    // Shifting every specific energy by the same constant changes neither the
    // interface residual nor chemistry heat when total mass is conserved.
    auto shiftedSolid = solid;
    auto shiftedGas = gas;
    auto shiftedPore = pore;
    constexpr double shift = 7e6;
    shiftedSolid[0].formationEnergy += shift;
    for (double& h : shiftedGas.formation) h += shift;
    for (std::size_t s=0; s<3; ++s) {
        shiftedPore.transportEnergy[s] += pore.transport[s]*shift;
        shiftedPore.sweptEnergy[s] += pore.sweep[s]*shift;
    }
    near(d::dryEnergyBalance(flux, shiftedPore, shiftedSolid, shiftedGas, energy).residual,
        balance.residual, "energy-reference invariance");
    auto elementalSolid = solid;
    auto elementalGas = gas;
    constexpr double carbonReference = 3e6, oxygenReference = -2e6;
    elementalSolid[0].formationEnergy += carbonReference;
    elementalGas.formation[0] += oxygenReference;
    elementalGas.formation[1] += (12*carbonReference+16*oxygenReference)/28;
    near(d::dryEnergyBalance(flux, pore, elementalSolid, elementalGas, energy).residual,
        balance.residual, "independent carbon/oxygen energy-reference invariance");

    // Analytic nonporous sublimation limit: required heat = m*(h_g-h_s),
    // with h_s=e_s+p/rho_s. This catches double-counted moving-face p work.
    d::SurfaceProduction<1, 3> sublimation{};
    sublimation.condensed[0] = -0.02;
    sublimation.gas[0] = 0.02;
    d::PoreSurfaceFlux<3> emptyPore;
    auto sublimationFlux = d::congruentDryInterfaceFlux(sublimation, emptyPore, solid, 0);
    d::DryEnergyInput latent{};
    latent.wallTemperature = 1000;
    latent.gasPressure = 1e5;
    latent.radiationIntoInterface = 0.02*(gas.enthalpy(0, 1000)
        -solid[0].internalEnergy(1000)-latent.gasPressure/solid[0].density);
    near(d::dryEnergyBalance(sublimationFlux, emptyPore, solid, gas, latent).residual,
        0, "sublimation latent heat");

    // Zero net species transport can still exchange energy with different
    // counterflow donor states. Never infer energy by dividing net mass flux.
    auto counterflow = emptyPore;
    counterflow.transport[2] = 0;
    counterflow.transportEnergy[2] = 0.1*(gas.enthalpy(2, 900)-gas.enthalpy(2, 800));
    const auto counterFlux = d::congruentDryInterfaceFlux(inert, counterflow, solid, 0.25);
    d::DryEnergyInput neutral{};
    neutral.wallTemperature = 1000;
    near(counterFlux.gasMass, 0, "counterflow net mass");
    near(d::dryEnergyBalance(counterFlux, counterflow, solid, gas, neutral).residual,
        -9000, "counterflow energy survives cancellation");

    // Recession exposes pore inventory: its energy is e, not h. The pressure
    // correction for both solid and pore swept volume is supplied only once.
    auto porousSweep = emptyPore;
    constexpr double phi = 0.25, poreRho = 2;
    auto sweptFlux = d::congruentDryInterfaceFlux(sublimation, porousSweep, solid, phi);
    porousSweep.sweep[2] = phi*poreRho*sweptFlux.recessionSpeed;
    const double poreE = gas.enthalpy(2, 1000)-1e5/poreRho;
    porousSweep.sweptEnergy[2] = porousSweep.sweep[2]*poreE;
    sweptFlux = d::congruentDryInterfaceFlux(sublimation, porousSweep, solid, phi);
    latent.radiationIntoInterface = 0.02*(gas.enthalpy(0, 1000)
        -solid[0].internalEnergy(1000)-latent.gasPressure/solid[0].density);
    near(d::dryEnergyBalance(sweptFlux, porousSweep, solid, gas, latent).residual,
        0, "porous swept pressure cancellation", 1e-10);

    // Separate O/O2/inert fixture, 2 O -> O2 (NOT the earlier O/CO fixture).
    // Catalysis transports species but neither removes material nor creates
    // net blowing; formation enthalpy differences supply the reaction heat.
    GasThermo catalyticGas;
    catalyticGas.formation = {{1.5e7, 0, 0}};
    d::SurfaceProduction<1, 3> catalytic{};
    catalytic.gas = {{-0.032, 0.032, 0}};
    const auto catalyticFlux = d::congruentDryInterfaceFlux(catalytic, emptyPore, solid, 0);
    near(catalyticFlux.gasMass, 0, "catalysis net mass");
    near(catalyticFlux.recessionSpeed, 0, "catalysis recession");
    near(catalytic.gas[0]+catalytic.gas[1], 0, "catalysis oxygen-element balance");
    near(d::dryEnergyBalance(catalyticFlux, emptyPore, solid, catalyticGas, neutral).residual,
        0.032*(catalyticGas.enthalpy(1, 1000)-catalyticGas.enthalpy(0, 1000)),
        "reaction heat once");

    std::cout << "dry surface: kinetics, elements, signed species, recession, ALE energy, "
                 "reference invariance, latent heat, pore counterflow and sweep passed\n";
}
