#ifndef CHMT_DRY_BULK_RESPONSE_H
#define CHMT_DRY_BULK_RESPONSE_H
#include "Types.H"
#include <cmath>
#include <limits>
#include <vector>

namespace chmt { namespace dry {



template<std::size_t Nc, std::size_t Ns, class GasThermo>
CHMT_DRY_HD CaloricState caloricState(const MaterialState<Nc,Ns>& q,
    const std::array<CondensedThermo,Nc>& solid, const GasThermo& gas, ThermalReal T)
{
    CaloricState out;
    for(std::size_t c=0;c<Nc;++c) {
        out.energy+=ThermalReal(q.condensedMass[c])*solid[c].internalEnergy(T);
        out.capacity+=ThermalReal(q.condensedMass[c])*solid[c].heatCapacity(T);
    }
    for(std::size_t s=0;s<Ns;++s) if(q.poreMass[s]!=0) {
        out.energy+=ThermalReal(q.poreMass[s])*ThermalReal(gas.internalEnergy(s,T));
        out.capacity+=ThermalReal(q.poreMass[s])*ThermalReal(gas.heatCapacityCv(s,T));
    }
    return out;
}





struct TemperatureSolve {
    ThermalReal temperature=0, residual=0;
    int iterations=0;
    bool converged=false;
};
template<std::size_t Nc, std::size_t Ns, class GasThermo>
CHMT_DRY_HD TemperatureSolve recoverTemperature(const MaterialState<Nc,Ns>& q,
    const std::array<CondensedThermo,Nc>& solid, const GasThermo& gas,
    ThermalReal initialTemperature, ThermalReal absoluteTolerance=1e-10f,
    ThermalReal relativeTolerance=std::numeric_limits<ThermalReal>::epsilon(), int maximumIterations=30)
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
CHMT_DRY_HD MaterialPrimitive primitive(const MaterialState<Nc,Ns>& q, PhysicalReal volume,
    PhysicalReal temperature, const std::array<CondensedThermo,Nc>& solid,
    const GasThermo& gas, PhysicalReal poreGasConductivity)
{
    MaterialPrimitive p;
    p.temperature=temperature;
    p.porosity=1;
    PhysicalReal mass=0, massR=0;
    for(std::size_t c=0;c<Nc;++c) if(q.condensedMass[c]!=0) {
        const PhysicalReal fraction=q.condensedMass[c]/(solid[c].density*volume);
        p.porosity-=fraction;
        p.conductivity+=ThermalReal(fraction)*solid[c].conductivity;
    }
    for(std::size_t s=0;s<Ns;++s) {
        mass+=q.poreMass[s];massR+=q.poreMass[s]*gas.gasConstant(s);
    }


    if(mass!=0) {
        p.poreDensity=mass/(p.porosity*volume);
        p.pressure=massR*temperature/(p.porosity*volume);
        p.conductivity+=ThermalReal(p.porosity)*ThermalReal(poreGasConductivity);
    }
    return p;
}

template<std::size_t Nc, std::size_t Ns>
struct Reaction {
    Stoichiometry<Nc,Ns> production;
    std::array<PhysicalReal,Nc> condensedOrder{};
    std::array<PhysicalReal,Ns> poreOrder{};
    PhysicalReal preExponential=0, temperaturePower=0, activationEnergy=0;
};



template<std::size_t Nc, std::size_t Ns>
struct ReactionView {
    const Reaction<Nc,Ns>* data=nullptr;
    std::size_t count=0;
    CHMT_DRY_HD const Reaction<Nc,Ns>* begin()const{return data;}
    CHMT_DRY_HD const Reaction<Nc,Ns>* end()const{return count?data+count:data;}
};









template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD PhysicalReal reactionRate(const Reaction<Nc,Ns>& reaction,
    const std::array<PhysicalReal,Nc>& condensedConcentration,
    const std::array<PhysicalReal,Ns>& poreConcentration, PhysicalReal temperature)
{
    constexpr PhysicalReal Ru=8.31446261815324f;
    PhysicalReal rate=reaction.preExponential*std::pow(temperature,reaction.temperaturePower)
        *std::exp(-reaction.activationEnergy/(Ru*temperature));
    for(std::size_t c=0;c<Nc;++c) if(reaction.condensedOrder[c]!=0)
        rate*=std::pow(condensedConcentration[c],reaction.condensedOrder[c]);
    for(std::size_t s=0;s<Ns;++s) if(reaction.poreOrder[s]!=0)
        rate*=std::pow(poreConcentration[s],reaction.poreOrder[s]);
    return rate;
}

template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD MaterialState<Nc,Ns> reactionSourceFromConcentrations(ReactionView<Nc,Ns> network,
    const std::array<PhysicalReal,Nc>& condensedConcentration,
    const std::array<PhysicalReal,Ns>& poreConcentration,
    PhysicalReal temperature, PhysicalReal volume)
{
    MaterialState<Nc,Ns> source;
    for(const auto& reaction:network) {
        const PhysicalReal extentRate=volume*reactionRate(reaction,
            condensedConcentration,poreConcentration,temperature);
        for(std::size_t c=0;c<Nc;++c)
            source.condensedMass[c]+=reaction.production.condensed[c]*extentRate;
        for(std::size_t s=0;s<Ns;++s)
            source.poreMass[s]+=reaction.production.gas[s]*extentRate;
    }

    return source;
}






template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD MaterialState<Nc,Ns> reactionSource(const MaterialState<Nc,Ns>& q,
    ReactionView<Nc,Ns> network, PhysicalReal volume,
    const std::array<CondensedThermo,Nc>& solid, PhysicalReal temperature)
{
    std::array<PhysicalReal,Nc> condensedConcentration{};
    std::array<PhysicalReal,Ns> poreConcentration{};
    PhysicalReal poreVolume=volume;
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






template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD PhysicalReal reactionEuler(MaterialState<Nc,Ns>& q,
    const MaterialState<Nc,Ns>& source, PhysicalReal dt)
{
    PhysicalReal fraction=1;
    for(std::size_t c=0;c<Nc;++c) if(source.condensedMass[c]<0) {
        const PhysicalReal available=q.condensedMass[c]/(-dt*source.condensedMass[c]);
        if(available<fraction)fraction=available;
    }
    for(std::size_t s=0;s<Ns;++s) if(source.poreMass[s]<0) {
        const PhysicalReal available=q.poreMass[s]/(-dt*source.poreMass[s]);
        if(available<fraction)fraction=available;
    }





    for(std::size_t c=0;c<Nc;++c) {
        const bool limiting=fraction<1 && source.condensedMass[c]<0
            && q.condensedMass[c]/(-dt*source.condensedMass[c])==fraction;
        if(limiting)q.condensedMass[c]=0;
        else q.condensedMass[c]+=dt*fraction*source.condensedMass[c];
    }
    for(std::size_t s=0;s<Ns;++s) {
        const bool limiting=fraction<1 && source.poreMass[s]<0
            && q.poreMass[s]/(-dt*source.poreMass[s])==fraction;
        if(limiting)q.poreMass[s]=0;
        else q.poreMass[s]+=dt*fraction*source.poreMass[s];
    }
    return fraction;
}





template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD void applyExtent(MaterialState<Nc,Ns>& q,
    const Stoichiometry<Nc,Ns>& production, PhysicalReal extent)
{
    for(std::size_t c=0;c<Nc;++c)q.condensedMass[c]+=extent*production.condensed[c];
    for(std::size_t s=0;s<Ns;++s)q.poreMass[s]+=extent*production.gas[s];
}



template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD PhysicalReal massResidual(const Stoichiometry<Nc,Ns>& reaction)
{
    PhysicalReal residual=0;
    for(PhysicalReal x:reaction.condensed)residual+=x;
    for(PhysicalReal x:reaction.gas)residual+=x;
    return residual;
}
template<std::size_t Nc, std::size_t Ns, std::size_t Ne>
CHMT_DRY_HD std::array<PhysicalReal,Ne> elementResidual(const Stoichiometry<Nc,Ns>& reaction,
    const std::array<std::array<PhysicalReal,Ne>,Nc>& solidElementMassFractions,
    const std::array<std::array<PhysicalReal,Ne>,Ns>& gasElementMassFractions)
{
    std::array<PhysicalReal,Ne> residual{};
    for(std::size_t e=0;e<Ne;++e) {
        for(std::size_t c=0;c<Nc;++c)
            residual[e]+=reaction.condensed[c]*solidElementMassFractions[c][e];
        for(std::size_t s=0;s<Ns;++s)
            residual[e]+=reaction.gas[s]*gasElementMassFractions[s][e];
    }
    return residual;
}

template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSourceFromConcentrations(const std::vector<Reaction<Nc,Ns>>& network,
    const std::array<PhysicalReal,Nc>& condensedConcentration,
    const std::array<PhysicalReal,Ns>& poreConcentration,PhysicalReal temperature,PhysicalReal volume)
{
    return reactionSourceFromConcentrations(ReactionView<Nc,Ns>{network.data(),network.size()},
        condensedConcentration,poreConcentration,temperature,volume);
}
template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSource(const MaterialState<Nc,Ns>& q,
    const std::vector<Reaction<Nc,Ns>>& network,PhysicalReal volume,
    const std::array<CondensedThermo,Nc>& solid,PhysicalReal temperature)
{
    return reactionSource(q,ReactionView<Nc,Ns>{network.data(),network.size()},volume,solid,temperature);
}
} }
#endif
