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



    double prefactor = 0;
    double temperaturePower = 0;
    double activationEnergy = 0;
    std::array<double, Nc> condensedOrder{};
    std::array<double, Ns> gasOrder{};
};

template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD double surfaceExtentRate
(
    const SurfaceMassAction<Nc, Ns>& reaction,
    double temperature,
    const std::array<double, Nc>& condensedConcentration,
    const std::array<double, Ns>& gasPartialDensity
)
{
    if (reaction.prefactor == 0) return 0;
    constexpr double Ru = 8.31446261815324;
    double logRate = std::log(reaction.prefactor)
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
    std::array<double, Nc> condensed{};
    std::array<double, Ns> gas{};
    double massResidual = 0;
};

template<std::size_t Nc, std::size_t Ns, std::size_t Nr>
CHMT_DRY_HD SurfaceProduction<Nc, Ns> surfaceProduction
(
    const std::array<SurfaceMassAction<Nc, Ns>, Nr>& reactions,
    double temperature,
    const std::array<double, Nc>& condensedConcentration,
    const std::array<double, Ns>& gasPartialDensity
)
{
    SurfaceProduction<Nc, Ns> out;


    for (const auto& reaction : reactions) {
        const double rate = surfaceExtentRate(reaction, temperature,
            condensedConcentration, gasPartialDensity);
        for (std::size_t c=0; c<Nc; ++c)
            out.condensed[c] += rate*reaction.stoichiometry.condensed[c];
        for (std::size_t s=0; s<Ns; ++s)
            out.gas[s] += rate*reaction.stoichiometry.gas[s];
    }
    for (double source : out.condensed) out.massResidual += source;
    for (double source : out.gas) out.massResidual += source;
    return out;
}

template<std::size_t Ns>
struct PoreSurfaceFlux {



    std::array<double, Ns> transport{};
    std::array<double, Ns> sweep{};




    std::array<double, Ns> transportEnergy{};
    std::array<double, Ns> sweptEnergy{};
};

template<std::size_t Nc, std::size_t Ns>
struct DryInterfaceFlux {
    std::array<double, Nc> condensedConsumption{};
    std::array<double, Ns> gasSpecies{};
    double gasMass = 0;
    double recessionSpeed = 0;
    double normalInterfaceSpeed = 0;
    double reactionMassResidual = 0;
};

template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD DryInterfaceFlux<Nc, Ns> congruentDryInterfaceFlux
(
    const SurfaceProduction<Nc, Ns>& production,
    const PoreSurfaceFlux<Ns>& pore,
    const std::array<CondensedThermo, Nc>& condensedThermo,
    double porosity
)
{
    DryInterfaceFlux<Nc, Ns> out;
    double consumedVolume = 0;
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
CHMT_DRY_HD std::array<double, Ns> surfaceSpeciesResidual
(
    const std::array<double, Ns>& gasSpeciesFlux,
    const std::array<double, Ns>& wallMassFraction,
    const std::array<double, Ns>& gasDiffusiveFlux,
    double gasMassFlux
)
{
    std::array<double, Ns> residual{};
    for (std::size_t s=0; s<Ns; ++s)
        residual[s] = gasMassFlux*wallMassFraction[s]
            + gasDiffusiveFlux[s] - gasSpeciesFlux[s];

    return residual;
}

struct DryEnergyInput {
    double wallTemperature = 0;
    double gasPressure = 0;
    double gasSpecificKineticEnergy = 0;
    double gasConductiveFlux = 0;
    double gasTractionPower = 0;
    double solidConductiveFlux = 0;
    double radiationIntoInterface = 0;
};

struct DryEnergyBalance {
    double gasThermochemical = 0;
    double gasKinetic = 0;
    double gasPressureWork = 0;
    double gasEnergy = 0;
    double solidDonorEnergy = 0;
    double poreTransportEnergy = 0;
    double poreSweptEnergy = 0;
    double materialEnergy = 0;
    double residual = 0;
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
        out.gasThermochemical += flux.gasSpecies[s]
            *gasThermo.enthalpy(s, in.wallTemperature);
        out.poreTransportEnergy += pore.transportEnergy[s];
        out.poreSweptEnergy += pore.sweptEnergy[s];
    }
    for (std::size_t c=0; c<Nc; ++c)
        out.solidDonorEnergy += flux.condensedConsumption[c]
            *condensedThermo[c].internalEnergy(in.wallTemperature);
    out.gasKinetic = flux.gasMass*in.gasSpecificKineticEnergy;


    out.gasPressureWork = in.gasPressure*flux.normalInterfaceSpeed;
    out.gasEnergy = out.gasThermochemical + out.gasKinetic + out.gasPressureWork
        + in.gasConductiveFlux - in.gasTractionPower;
    out.materialEnergy = out.solidDonorEnergy + out.poreTransportEnergy
        + out.poreSweptEnergy + in.solidConductiveFlux;
    out.residual = out.gasEnergy - out.materialEnergy - in.radiationIntoInterface;


    return out;
}

}}
#endif
