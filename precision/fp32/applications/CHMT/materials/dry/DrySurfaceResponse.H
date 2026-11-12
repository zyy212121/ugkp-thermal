#ifndef CHMT_DRY_SURFACE_RESPONSE_H
#define CHMT_DRY_SURFACE_RESPONSE_H

#include "Types.H"
#include <array>
#include <cmath>
#include <cstddef>

namespace chmt { namespace dry {

// Dry, zero-storage surface balance; normal points from material into gas.
// Every flux is per CURRENT physical material-interface area. A mesh adapter, not this
// constitutive kernel, maps areas/normals to gas faces conservatively.
// This CPU kernel evaluates constitutive rates/residuals, not gas equations,
// time stepping, inventory limiting, or a new gas wall closure. Its caller owns
// the coupled wall temperature/composition solve and the accepted flux packet.
// Mathematical reference (not copied implementation): Mutation++ GSI balances,
// https://github.com/mutationpp/Mutationpp/blob/master/docs/input-files.md#gas-surface-interaction
// No liquid film, equilibrium B-prime table, or empirical blowing correction.

template<std::size_t Nc, std::size_t Ns>
struct SurfaceMassAction {
    Stoichiometry<Nc, Ns> stoichiometry{};
    // Signed stoichiometry is kg/mol extent; production positive in each phase.
    // A*T^b*exp(-Ea/(Ru*T))*product(concentration^order) is mol/(m2 s).
    // Both concentrations below are kg/m3, so A's units depend on the orders.
    PhysicalReal prefactor = 0;
    PhysicalReal temperaturePower = 0;
    PhysicalReal activationEnergy = 0; // J/mol
    std::array<PhysicalReal, Nc> condensedOrder{};
    std::array<PhysicalReal, Ns> gasOrder{};
};

template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD PhysicalReal surfaceExtentRate
(
    const SurfaceMassAction<Nc, Ns>& reaction,
    PhysicalReal temperature,
    const std::array<PhysicalReal, Nc>& condensedConcentration,
    const std::array<PhysicalReal, Ns>& gasPartialDensity
)
{
    if (reaction.prefactor == 0) return 0;
    constexpr PhysicalReal Ru = 8.31446261815324f;
    PhysicalReal logRate = std::log(reaction.prefactor)
        + reaction.temperaturePower*std::log(temperature)
        - reaction.activationEnergy/(Ru*temperature);
    for (std::size_t c=0; c<Nc; ++c) {
        if (reaction.condensedOrder[c] == 0) continue;
        if (condensedConcentration[c] == 0 && reaction.condensedOrder[c] > 0)
            return 0;
        logRate += reaction.condensedOrder[c]*std::log(condensedConcentration[c]);
    }
    for (std::size_t s=0; s<Ns; ++s) {
        if (reaction.gasOrder[s] == 0) continue;
        if (gasPartialDensity[s] == 0 && reaction.gasOrder[s] > 0) return 0;
        logRate += reaction.gasOrder[s]*std::log(gasPartialDensity[s]);
    }
    return std::exp(logRate);
}

template<std::size_t Nc, std::size_t Ns>
struct SurfaceProduction {
    std::array<PhysicalReal, Nc> condensed{}; // signed production, kg/(m2 s)
    std::array<PhysicalReal, Ns> gas{};       // includes negative oxidant consumption
    PhysicalReal massResidual = 0;           // sum of all phase sources, diagnostic
};

template<std::size_t Nc, std::size_t Ns, std::size_t Nr>
CHMT_DRY_HD SurfaceProduction<Nc, Ns> surfaceProduction
(
    const std::array<SurfaceMassAction<Nc, Ns>, Nr>& reactions,
    PhysicalReal temperature,
    const std::array<PhysicalReal, Nc>& condensedConcentration,
    const std::array<PhysicalReal, Ns>& gasPartialDensity
)
{
    SurfaceProduction<Nc, Ns> out;
    // A reversible process is supplied as separately calibrated forward/reverse
    // channels. No equilibrium composition or reaction enthalpy is invented.
    for (const auto& reaction : reactions) {
        const PhysicalReal rate = surfaceExtentRate(reaction, temperature,
            condensedConcentration, gasPartialDensity);
        for (std::size_t c=0; c<Nc; ++c)
            out.condensed[c] += rate*reaction.stoichiometry.condensed[c];
        for (std::size_t s=0; s<Ns; ++s)
            out.gas[s] += rate*reaction.stoichiometry.gas[s];
    }
    for (PhysicalReal source : out.condensed) out.massResidual += source;
    for (PhysicalReal source : out.gas) out.massResidual += source;
    return out;
}

template<std::size_t Ns>
struct PoreSurfaceFlux {
    // Signed mass fluxes per current physical interface area. Transport is physical Darcy
    // advection plus diffusion; sweep is solely the moving-face inventory flux.
    // Keep them separate: equal/opposite mass need not cancel donor energy.
    std::array<PhysicalReal, Ns> transport{};
    std::array<PhysicalReal, Ns> sweep{};
    // Caller sums channel energy BEFORE combining signed mass fluxes, using
    // the SAME thermo as the gas. Physical Darcy/diffusion carries h; geometric
    // sweep carries stored e. Separate donor temperatures are allowed even if
    // opposing Darcy/diffusion mass fluxes cancel exactly. Units W/m2.
    std::array<ThermalReal, Ns> transportEnergy{};
    std::array<ThermalReal, Ns> sweptEnergy{};
};

template<std::size_t Nc, std::size_t Ns>
struct DryInterfaceFlux {
    std::array<PhysicalReal, Nc> condensedConsumption{}; // positive loss; negative growth
    std::array<PhysicalReal, Ns> gasSpecies{}; // total signed flux to gas, kg/(m2 s)
    PhysicalReal gasMass = 0;
    PhysicalReal recessionSpeed = 0; // m/s into stationary rigid material
    PhysicalReal normalInterfaceSpeed = 0; // m/s along outward normal, negative recession
    PhysicalReal reactionMassResidual = 0;
};

template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD DryInterfaceFlux<Nc, Ns> congruentDryInterfaceFlux
(
    const SurfaceProduction<Nc, Ns>& production,
    const PoreSurfaceFlux<Ns>& pore,
    const std::array<CondensedThermo, Nc>& condensedThermo,
    PhysicalReal porosity
)
{
    DryInterfaceFlux<Nc, Ns> out;
    PhysicalReal consumedVolume = 0;
    for (std::size_t c=0; c<Nc; ++c) {
        out.condensedConsumption[c] = -production.condensed[c];
        if (out.condensedConsumption[c] != 0)
            consumedVolume += out.condensedConsumption[c]/condensedThermo[c].density;
    }
    // Consumed true solid volume / packing fraction gives swept bulk volume.
    // Geometry closure: ONE exposed effective condensed phase, or congruent
    // recession of a mixture (loss_c = rho_c*alpha_c*recessionSpeed for every c).
    // Multiphase pyrolysis belongs in the bulk model. Arbitrary selective
    // multi-solid front chemistry is NOT closed by this geometry model: it
    // would require an additional surface composition/storage evolution law.
    // surfaceProduction above remains a generic stoichiometric flux routine.
    // Pyrolysis arrival is NOT solid ablation and must not set recession speed.
    out.recessionSpeed = consumedVolume == 0 ? 0 : consumedVolume/(1-porosity);
    out.normalInterfaceSpeed = -out.recessionSpeed;
    for (std::size_t s=0; s<Ns; ++s) {
        out.gasSpecies[s] = pore.transport[s] + pore.sweep[s] + production.gas[s];
        out.gasMass += out.gasSpecies[s];
    }
    out.reactionMassResidual = production.massResidual;
    return out;
}

template<std::size_t Ns>
CHMT_DRY_HD std::array<PhysicalReal, Ns> surfaceSpeciesResidual
(
    const std::array<PhysicalReal, Ns>& gasSpeciesFlux,
    const std::array<PhysicalReal, Ns>& wallMassFraction,
    const std::array<PhysicalReal, Ns>& gasDiffusiveFlux,
    PhysicalReal gasMassFlux
)
{
    std::array<PhysicalReal, Ns> residual{};
    for (std::size_t s=0; s<Ns; ++s)
        residual[s] = gasMassFlux*wallMassFraction[s]
            + gasDiffusiveFlux[s] - gasSpeciesFlux[s];
    // Signed source flux is never normalized into a fictitious composition.
    return residual;
}

struct DryEnergyInput {
    ThermalReal wallTemperature = 0;
    ThermalReal gasPressure = 0;
    ThermalReal gasSpecificKineticEnergy = 0; // |u_g|^2/2, laboratory frame
    ThermalReal gasConductiveFlux = 0; // Fourier heat along outward normal, W/m2
    ThermalReal gasTractionPower = 0;  // (tau.n).u_g, subtracted from outward flux
    ThermalReal solidConductiveFlux = 0; // from material into interface, W/m2
    ThermalReal radiationIntoInterface = 0; // absorbed minus emitted, W/m2
};

struct DryEnergyBalance {
    ThermalReal gasThermochemical = 0;
    ThermalReal gasKinetic = 0;
    ThermalReal gasPressureWork = 0;
    ThermalReal gasEnergy = 0; // full outward laboratory energy flux
    ThermalReal solidDonorEnergy = 0;
    ThermalReal poreTransportEnergy = 0;
    ThermalReal poreSweptEnergy = 0;
    ThermalReal materialEnergy = 0;
    ThermalReal residual = 0; // outward gas - incoming material - absorbed radiation
};

template<std::size_t Nc, std::size_t Ns, class GasThermo>
CHMT_DRY_HD DryEnergyBalance dryEnergyBalance
(
    const DryInterfaceFlux<Nc, Ns>& flux,
    const PoreSurfaceFlux<Ns>& pore,
    const std::array<CondensedThermo, Nc>& condensedThermo,
    const GasThermo& gasThermo,
    const DryEnergyInput& in
)
{
    DryEnergyBalance out;
    for (std::size_t s=0; s<Ns; ++s) {
        out.gasThermochemical += ThermalReal(flux.gasSpecies[s])
            *ThermalReal(gasThermo.enthalpy(s, in.wallTemperature));
        out.poreTransportEnergy += pore.transportEnergy[s];
        out.poreSweptEnergy += pore.sweptEnergy[s];
    }
    for (std::size_t c=0; c<Nc; ++c)
        out.solidDonorEnergy += ThermalReal(flux.condensedConsumption[c])
            *condensedThermo[c].internalEnergy(in.wallTemperature);
    out.gasKinetic = ThermalReal(flux.gasMass)*in.gasSpecificKineticEnergy;
    // rho(u-vI)h + p*vI is the moving-control-surface total-energy flux.
    // Solid is stationary: its swept inventory carries e, not a second p/rho.
    out.gasPressureWork = in.gasPressure*ThermalReal(flux.normalInterfaceSpeed);
    out.gasEnergy = out.gasThermochemical + out.gasKinetic + out.gasPressureWork
        + in.gasConductiveFlux - in.gasTractionPower;
    out.materialEnergy = out.solidDonorEnergy + out.poreTransportEnergy
        + out.poreSweptEnergy + in.solidConductiveFlux;
    out.residual = out.gasEnergy - out.materialEnergy - in.radiationIntoInterface;
    // Formation energies in h/e already account for surface reaction heat.
    // There is deliberately no additional Q_reaction or empirical blowing term.
    return out;
}

}} // namespace chmt::dry
#endif
