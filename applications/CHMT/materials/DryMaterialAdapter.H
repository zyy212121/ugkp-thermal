#ifndef CHMT_MATERIALS_DRYMATERIALADAPTER_H
#define CHMT_MATERIALS_DRYMATERIALADAPTER_H
#include "materials/ReactionStepControl.H"
#include "materials/dry/BulkResponse.H"
#include "materials/dry/PoreTransport.H"

namespace chmt { namespace dryAdapter {


// Host Nc is unchanged (currently two condensed inventories); this is not a
// projection/truncation adapter for higher-dimensional benchmark mechanisms.
using State = dry::MaterialState<Nc,Ns>;
struct GasCalorics {
    const PhysicsConfig& p;
    CHMT_HD double internalEnergy(std::size_t s,double T)const{return speciesE(p.species[s],T);}
    CHMT_HD double heatCapacityCv(std::size_t s,double T)const{return speciesCv(p.species[s],T);}
    CHMT_HD double enthalpy(std::size_t s,double T)const{return speciesH(p.species[s],T);}
    CHMT_HD double gasConstant(std::size_t s)const{return p.species[s].R;}
};
CHMT_HD inline State state(const SolidQ& q) {
    State out;out.energy=q.energy;
    for(int c=0;c<Nc;++c)out.condensedMass[c]=q.condensed[c];
    for(int s=0;s<Ns;++s)out.poreMass[s]=q.pore[s];
    return out;
}
CHMT_HD inline std::array<dry::CondensedThermo,Nc> condensed(const PhysicsConfig& p) {
    std::array<dry::CondensedThermo,Nc> out{};
    for(int c=0;c<Nc;++c){const auto& a=p.condensed[c];auto& b=out[c];
        b.density=a.rho;b.cp0=a.cp0;b.cp1=a.cp1;b.formationEnergy=a.e0;b.conductivity=a.conductivity;}
    return out;
}
CHMT_HD inline dry::Reaction<Nc,Ns> reaction(const chmt::Reaction& a) {
    dry::Reaction<Nc,Ns> b;b.preExponential=a.A;
    b.temperaturePower=a.temperaturePower;b.activationEnergy=a.activationEnergy;
    for(int c=0;c<Nc;++c){b.condensedOrder[c]=a.order[c];b.production.condensed[c]=a.condensedNu[c];}
    for(int s=0;s<Ns;++s)b.production.gas[s]=a.gasNu[s];
    return b;
}
inline std::vector<dry::Reaction<Nc,Ns>> mechanism(const PhysicsConfig& p) {
    std::vector<dry::Reaction<Nc,Ns>> result;result.reserve(p.nReactions);
    for(int r=0;r<p.nReactions;++r)result.push_back(reaction(p.reactions[r]));
    return result;
}
struct CoreReactionRates {
    CHMT_HD double rate(const chmt::Reaction& input,const SolidQ& q,double V,double T)const {
        std::array<double,Nc> c{};std::array<double,Ns> g{};
        for(int k=0;k<Nc;++k)c[k]=q.condensed[k]/V;
        return V*dry::reactionRate(reaction(input),c,g,T);
    }
    CHMT_HD double derivative(const chmt::Reaction& input,int k,const SolidQ& q,double V,double T,double value)const {
        const double order=input.order[k];if(order==0)return 0;
        if(q.condensed[k]>0)return value*order/q.condensed[k];
        if(order!=1)return 0;
        auto unit=q;unit.condensed[k]=V;
        return rate(input,unit,V,T)/V;
    }
};
CHMT_HD inline bool reactionStep(const SolidQ& base,Real volume,const PhysicsConfig& p,
    dry::ReactionView<Nc,Ns> network,Real maximumStep,Real cfl,
    SolidQ& output,Real& step,bool& active,bool* numericalFailure) {


    ReactionStepControl control;
    if(!reactionStepControl(base,volume,p,cfl,control,numericalFailure,CoreReactionRates{}))return false;
    active=control.active;step=active?minValue(maximumStep,control.dtLimit):maximumStep;
    output=base;if(!active)return true;
    auto q=state(base);
    const auto source=dry::reactionSource(q,network,volume,condensed(p),control.temperature);
    const Real fraction=dry::reactionEuler(q,source,step);
    for(int c=0;c<Nc;++c)output.condensed[c]=q.condensedMass[c];
    for(int s=0;s<Ns;++s)output.pore[s]=q.poreMass[s];
    std::array<double,Nc> concentration{};std::array<double,Ns> poreConcentration{};
    for(int c=0;c<Nc;++c)concentration[c]=base.condensed[c]/volume;
    for(std::size_t r=0;r<network.count;++r)
        output.progress[r]+=step*fraction*volume*dry::reactionRate(network.data[r],concentration,poreConcentration,control.temperature);
    MaterialPrimitive endpoint;
    if(!recoverMaterial(output,volume,p,endpoint,nullptr,-1,numericalFailure))return false;
    output.porosity=endpoint.porosity;return true;
}
inline bool reactionStep(const SolidQ& base,Real volume,const PhysicsConfig& p,
    const std::vector<dry::Reaction<Nc,Ns>>& network,Real maximumStep,Real cfl,
    SolidQ& output,Real& step,bool& active,bool* numericalFailure) {
    return reactionStep(base,volume,p,dry::ReactionView<Nc,Ns>{network.data(),network.size()},
        maximumStep,cfl,output,step,active,numericalFailure);
}
CHMT_HD inline dry::PoreFaceState<Ns> poreState(const SolidQ& q,const MaterialPrimitive& p) {
    dry::PoreFaceState<Ns> out;out.density=p.poreDensity;out.temperature=p.temperature;
    Real mass=0;for(int s=0;s<Ns;++s)mass+=q.pore[s];
    if(mass>0)for(int s=0;s<Ns;++s)out.massFraction[s]=q.pore[s]/mass;
    return out;
}
// Composition is undefined in an empty pore. Use the occupied-side limit,
// not a zero-sum composition; this preserves sum(J)=0 without inventing mass.
CHMT_HD inline void completePorePair(dry::PoreFaceState<Ns>& a,dry::PoreFaceState<Ns>& b) {
    if(a.density==0&&b.density>0)a.massFraction=b.massFraction;
    else if(b.density==0&&a.density>0)b.massFraction=a.massFraction;
    else if(a.density==0&&b.density==0){a.massFraction[0]=1;b.massFraction=a.massFraction;}
}
} }
#endif
