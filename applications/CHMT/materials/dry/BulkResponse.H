#ifndef CHMT_DRY_BULK_RESPONSE_H
#define CHMT_DRY_BULK_RESPONSE_H
#include "Types.H"
#include <cmath>
#include <vector>

namespace chmt { namespace dry {
// Volume-averaged LTE material balance as in PATO (Lachaud et al., JTHT 2014,
// https://pato.ac/wp-content/uploads/2017/06/Lachaud_2014_JTHT.pdf). A finite-rate source transfers constituent mass but
// does not add a separate reaction heat: formation energy is already in e.
template<std::size_t Nc, std::size_t Ns, class GasThermo>
CaloricState caloricState(const MaterialState<Nc,Ns>& q,
    const std::array<CondensedThermo,Nc>& solid, const GasThermo& gas, double T)
{
    CaloricState out;
    for(std::size_t c=0;c<Nc;++c) {
        out.energy+=q.condensedMass[c]*solid[c].internalEnergy(T);
        out.capacity+=q.condensedMass[c]*solid[c].heatCapacity(T);
    }
    for(std::size_t s=0;s<Ns;++s) if(q.poreMass[s]!=0) {
        out.energy+=q.poreMass[s]*gas.internalEnergy(s,T);
        out.capacity+=q.poreMass[s]*gas.heatCapacityCv(s,T);
    }
    return out;
}

// A numerical inversion only, not an admissibility/preflight policy. No
// clipping, rollback, or configuration rejection. Caller owns error handling.
// Initial T should be the previous LTE temperature. The residual is returned
// even on nonconvergence so existing diagnostics can report it unchanged.
struct TemperatureSolve {
    double temperature=0, residual=0;
    int iterations=0;
    bool converged=false;
};
template<std::size_t Nc, std::size_t Ns, class GasThermo>
TemperatureSolve recoverTemperature(const MaterialState<Nc,Ns>& q,
    const std::array<CondensedThermo,Nc>& solid, const GasThermo& gas,
    double initialTemperature, double absoluteTolerance=1e-10,
    double relativeTolerance=1e-12, int maximumIterations=30)
{
    TemperatureSolve out;
    out.temperature=initialTemperature;
    for(int k=0;k<=maximumIterations;++k) {
        const auto caloric=caloricState(q,solid,gas,out.temperature);
        out.residual=caloric.energy-q.energy;
        out.iterations=k;
        if(std::abs(out.residual)<=absoluteTolerance+
            relativeTolerance*std::abs(q.energy)) {out.converged=true;break;}
        if(k==maximumIterations)break;
        out.temperature-=out.residual/caloric.capacity;
    }
    return out;
}

template<std::size_t Nc, std::size_t Ns, class GasThermo>
MaterialPrimitive primitive(const MaterialState<Nc,Ns>& q, double volume,
    double temperature, const std::array<CondensedThermo,Nc>& solid,
    const GasThermo& gas, double poreGasConductivity)
{
    MaterialPrimitive p;
    p.temperature=temperature;
    p.porosity=1;
    double mass=0, massR=0;
    for(std::size_t c=0;c<Nc;++c) if(q.condensedMass[c]!=0) {
        const double fraction=q.condensedMass[c]/(solid[c].density*volume);
        p.porosity-=fraction;
        p.conductivity+=fraction*solid[c].conductivity;
    }
    for(std::size_t s=0;s<Ns;++s) {
        mass+=q.poreMass[s];massR+=q.poreMass[s]*gas.gasConstant(s);
    }
    // An empty pore carries neither pressure nor gas heat conduction. Positive
    // gas inventory with zero pore volume is not silently clipped here.
    if(mass!=0) {
        p.poreDensity=mass/(p.porosity*volume);
        p.pressure=massR*temperature/(p.porosity*volume);
        p.conductivity+=p.porosity*poreGasConductivity;
    }
    return p;
}

template<std::size_t Nc, std::size_t Ns>
struct Reaction {
    Stoichiometry<Nc,Ns> production;
    std::array<double,Nc> condensedOrder{};
    std::array<double,Ns> poreOrder{};
    double preExponential=0, temperaturePower=0, activationEnergy=0; // Ea J/mol
};

// Rate is extent/(m3 bulk s), matching production coefficients kg/extent.
// State-based pyrolysis uses MASS concentrations [kg/m3], just as existing
// CHMT data: A has units extent/(m3 s K^b (kg/m3)^sum(orders)).
// Explicit concentration callers may supply a calibrated molar mechanism, but
// must use its corresponding A/extent units; no automatic basis conversion.
// Gas-order extensions are still calibrated per BULK volume. A homogeneous
// pore-gas mechanism expressed per pore volume needs a separate phi conversion.
// An empty mechanism means frozen pore chemistry, not frozen species transport.
template<std::size_t Nc, std::size_t Ns>
double reactionRate(const Reaction<Nc,Ns>& reaction,
    const std::array<double,Nc>& condensedConcentration,
    const std::array<double,Ns>& poreConcentration, double temperature)
{
    constexpr double Ru=8.31446261815324;
    double rate=reaction.preExponential*std::pow(temperature,reaction.temperaturePower)
        *std::exp(-reaction.activationEnergy/(Ru*temperature));
    for(std::size_t c=0;c<Nc;++c) if(reaction.condensedOrder[c]!=0)
        rate*=std::pow(condensedConcentration[c],reaction.condensedOrder[c]);
    for(std::size_t s=0;s<Ns;++s) if(reaction.poreOrder[s]!=0)
        rate*=std::pow(poreConcentration[s],reaction.poreOrder[s]);
    return rate;
}

template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSourceFromConcentrations(const std::vector<Reaction<Nc,Ns>>& network,
    const std::array<double,Nc>& condensedConcentration,
    const std::array<double,Ns>& poreConcentration,
    double temperature, double volume)
{
    MaterialState<Nc,Ns> source;
    for(const auto& reaction:network) {
        const double extentRate=volume*reactionRate(reaction,
            condensedConcentration,poreConcentration,temperature);
        for(std::size_t c=0;c<Nc;++c)
            source.condensedMass[c]+=reaction.production.condensed[c]*extentRate;
        for(std::size_t s=0;s<Ns;++s)
            source.poreMass[s]+=reaction.production.gas[s]*extentRate;
    }
    // energy source is zero: e_formation is included in caloricState.
    return source;
}

// Assemble MASS concentrations [kg/m3] from stored inventories, preserving
// existing pyrolysis A, orders and mass-yield coefficients without requiring
// a fictitious resin molecular weight. Condensed concentration is per bulk
// volume; gas reactants use intrinsic pore partial density. Existing bulk
// pyrolysis has poreOrder=0 and is therefore unchanged by pore composition.
template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSource(const MaterialState<Nc,Ns>& q,
    const std::vector<Reaction<Nc,Ns>>& network, double volume,
    const std::array<CondensedThermo,Nc>& solid, double temperature)
{
    std::array<double,Nc> condensedConcentration{};
    std::array<double,Ns> poreConcentration{};
    double poreVolume=volume;
    for(std::size_t c=0;c<Nc;++c) {
        condensedConcentration[c]=q.condensedMass[c]/volume;
        if(q.condensedMass[c]!=0)poreVolume-=q.condensedMass[c]/solid[c].density;
    }
    for(const auto& reaction:network)
        for(std::size_t s=0;s<Ns;++s) if(reaction.poreOrder[s]!=0)
            poreConcentration[s]=q.poreMass[s]/poreVolume;
    return reactionSourceFromConcentrations(network,condensedConcentration,
        poreConcentration,temperature,volume);
}

// One conservative forward-Euler local chemistry step, using the same common
// mechanism-wide extent scaling as the existing material reaction step. This
// is not a gas timestep restriction, a retry, or independent species flooring.
// Return accepted extent fraction for existing warning/accuracy reporting.
// Subcycling/refinement of this first-order map is the time integrator's job.
template<std::size_t Nc, std::size_t Ns>
double reactionEuler(MaterialState<Nc,Ns>& q,
    const MaterialState<Nc,Ns>& source, double dt)
{
    double fraction=1;
    for(std::size_t c=0;c<Nc;++c) if(source.condensedMass[c]<0) {
        const double available=q.condensedMass[c]/(-dt*source.condensedMass[c]);
        if(available<fraction)fraction=available;
    }
    for(std::size_t s=0;s<Ns;++s) if(source.poreMass[s]<0) {
        const double available=q.poreMass[s]/(-dt*source.poreMass[s]);
        if(available<fraction)fraction=available;
    }
    for(std::size_t c=0;c<Nc;++c)
        q.condensedMass[c]+=dt*fraction*source.condensedMass[c];
    for(std::size_t s=0;s<Ns;++s)
        q.poreMass[s]+=dt*fraction*source.poreMass[s];
    return fraction;
}

// Apply an already time-integrated extent in the production coefficient basis.
// Numerical time integration
// chooses extents; this operation never clips species independently (which
// would destroy reaction mass/element/energy balance).
template<std::size_t Nc, std::size_t Ns>
void applyExtent(MaterialState<Nc,Ns>& q,
    const Stoichiometry<Nc,Ns>& production, double extent)
{
    for(std::size_t c=0;c<Nc;++c)q.condensedMass[c]+=extent*production.condensed[c];
    for(std::size_t s=0;s<Ns;++s)q.poreMass[s]+=extent*production.gas[s];
}



template<std::size_t Nc, std::size_t Ns>
double massResidual(const Stoichiometry<Nc,Ns>& reaction)
{
    double residual=0;
    for(double x:reaction.condensed)residual+=x;
    for(double x:reaction.gas)residual+=x;
    return residual;
}
template<std::size_t Nc, std::size_t Ns, std::size_t Ne>
std::array<double,Ne> elementResidual(const Stoichiometry<Nc,Ns>& reaction,
    const std::array<std::array<double,Ne>,Nc>& solidElementMassFractions,
    const std::array<std::array<double,Ne>,Ns>& gasElementMassFractions)
{
    std::array<double,Ne> residual{};
    for(std::size_t e=0;e<Ne;++e) {
        for(std::size_t c=0;c<Nc;++c)
            residual[e]+=reaction.condensed[c]*solidElementMassFractions[c][e];
        for(std::size_t s=0;s<Ns;++s)
            residual[e]+=reaction.gas[s]*gasElementMassFractions[s][e];
    }
    return residual;
}
} }
#endif
