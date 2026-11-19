#ifndef CHMT_DRY_BULK_RESPONSE_H
#define CHMT_DRY_BULK_RESPONSE_H
#include "Types.H"
#include <cmath>
#include <vector>

namespace chmt { namespace dry {



template<std::size_t Nc, std::size_t Ns, class GasThermo>
CHMT_DRY_HD CaloricState caloricState(const MaterialState<Nc,Ns>& q,
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





struct TemperatureSolve {
    double temperature=0, residual=0;
    int iterations=0;
    bool converged=false;
};
template<std::size_t Nc, std::size_t Ns, class GasThermo>
CHMT_DRY_HD TemperatureSolve recoverTemperature(const MaterialState<Nc,Ns>& q,
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
CHMT_DRY_HD MaterialPrimitive primitive(const MaterialState<Nc,Ns>& q, double volume,
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
    double preExponential=0, temperaturePower=0, activationEnergy=0;
};



template<std::size_t Nc, std::size_t Ns>
struct ReactionView {
    const Reaction<Nc,Ns>* data=nullptr;
    std::size_t count=0;
    CHMT_DRY_HD const Reaction<Nc,Ns>* begin()const{return data;}
    CHMT_DRY_HD const Reaction<Nc,Ns>* end()const{return count?data+count:data;}
};









template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD double reactionRate(const Reaction<Nc,Ns>& reaction,
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
CHMT_DRY_HD MaterialState<Nc,Ns> reactionSourceFromConcentrations(ReactionView<Nc,Ns> network,
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

    return source;
}






template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD MaterialState<Nc,Ns> reactionSource(const MaterialState<Nc,Ns>& q,
    ReactionView<Nc,Ns> network, double volume,
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






template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD double reactionEuler(MaterialState<Nc,Ns>& q,
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
    const Stoichiometry<Nc,Ns>& production, double extent)
{
    for(std::size_t c=0;c<Nc;++c)q.condensedMass[c]+=extent*production.condensed[c];
    for(std::size_t s=0;s<Ns;++s)q.poreMass[s]+=extent*production.gas[s];
}



template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD double massResidual(const Stoichiometry<Nc,Ns>& reaction)
{
    double residual=0;
    for(double x:reaction.condensed)residual+=x;
    for(double x:reaction.gas)residual+=x;
    return residual;
}
template<std::size_t Nc, std::size_t Ns, std::size_t Ne>
CHMT_DRY_HD std::array<double,Ne> elementResidual(const Stoichiometry<Nc,Ns>& reaction,
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

template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSourceFromConcentrations(const std::vector<Reaction<Nc,Ns>>& network,
    const std::array<double,Nc>& condensedConcentration,
    const std::array<double,Ns>& poreConcentration,double temperature,double volume)
{
    return reactionSourceFromConcentrations(ReactionView<Nc,Ns>{network.data(),network.size()},
        condensedConcentration,poreConcentration,temperature,volume);
}
template<std::size_t Nc, std::size_t Ns>
MaterialState<Nc,Ns> reactionSource(const MaterialState<Nc,Ns>& q,
    const std::vector<Reaction<Nc,Ns>>& network,double volume,
    const std::array<CondensedThermo,Nc>& solid,double temperature)
{
    return reactionSource(q,ReactionView<Nc,Ns>{network.data(),network.size()},volume,solid,temperature);
}
} }
#endif
