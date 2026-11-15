#ifndef CHMT_DRY_SURFACE_RESPONSE_H
#define CHMT_DRY_SURFACE_RESPONSE_H

#include "Types.H"
#include <array>
#include <cmath>
#include <cstddef>

namespace chmt { namespace dry {











template<std::size_t Nc, std::size_t Ns>
struct SurfaceMassAction {
    Stoichiometry<Nc, Ns> stoichiometry{};



    PhysicalReal prefactor = 0;
    PhysicalReal temperaturePower = 0;
    PhysicalReal activationEnergy = 0;
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
    std::array<PhysicalReal, Nc> condensed{};
    std::array<PhysicalReal, Ns> gas{};
    PhysicalReal massResidual = 0;
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



    std::array<PhysicalReal, Ns> transport{};
    std::array<PhysicalReal, Ns> sweep{};




    std::array<ThermalReal, Ns> transportEnergy{};
    std::array<ThermalReal, Ns> sweptEnergy{};
};

template<std::size_t Nc, std::size_t Ns>
struct DryInterfaceFlux {
    std::array<PhysicalReal, Nc> condensedConsumption{};
    std::array<PhysicalReal, Ns> gasSpecies{};
    PhysicalReal gasMass = 0;
    PhysicalReal recessionSpeed = 0;
    PhysicalReal normalInterfaceSpeed = 0;
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

    return residual;
}

struct DryEnergyInput {
    ThermalReal wallTemperature = 0;
    ThermalReal gasPressure = 0;
    ThermalReal gasSpecificKineticEnergy = 0;
    ThermalReal gasConductiveFlux = 0;
    ThermalReal gasTractionPower = 0;
    ThermalReal solidConductiveFlux = 0;
    ThermalReal radiationIntoInterface = 0;
};

struct DryEnergyBalance {
    ThermalReal gasThermochemical = 0;
    ThermalReal gasKinetic = 0;
    ThermalReal gasPressureWork = 0;
    ThermalReal gasEnergy = 0;
    ThermalReal solidDonorEnergy = 0;
    ThermalReal poreTransportEnergy = 0;
    ThermalReal poreSweptEnergy = 0;
    ThermalReal materialEnergy = 0;
    ThermalReal residual = 0;
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


    out.gasPressureWork = in.gasPressure*ThermalReal(flux.normalInterfaceSpeed);
    out.gasEnergy = out.gasThermochemical + out.gasKinetic + out.gasPressureWork
        + in.gasConductiveFlux - in.gasTractionPower;
    out.materialEnergy = out.solidDonorEnergy + out.poreTransportEnergy
        + out.poreSweptEnergy + in.solidConductiveFlux;
    out.residual = out.gasEnergy - out.materialEnergy - in.radiationIntoInterface;


    return out;
}

}}
#endif
