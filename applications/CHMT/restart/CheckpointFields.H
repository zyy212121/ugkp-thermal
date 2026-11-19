#pragma once
#include "../core/HostState.H"
#include "../../../common/GpuParticleCaloric.H"
#include <stdexcept>
namespace chmt { namespace checkpointFields {


template<std::size_t N>
std::array<int,N> order(const std::array<std::string,N>& saved,
    const std::array<std::string,N>& current,const char* quantity) {
    std::array<int,N> map{};
    for(std::size_t i=0;i<N;++i) {
        if(current[i].empty()&&saved[i].empty()){map[i]=int(i);continue;}
        int index=-1;
        for(std::size_t j=0;j<N;++j)if(saved[j]==current[i]) {
            if(index>=0)throw std::runtime_error(std::string("ambiguous checkpoint ")+quantity+" field "+current[i]);
            index=int(j);
        }
        if(index<0)throw std::runtime_error(std::string("missing checkpoint ")+quantity+" field "+current[i]);
        map[i]=index;
    }
    return map;
}
template<class R,std::size_t N> void reorder(R (&values)[N],const std::array<int,N>& map) {
    R old[N];for(std::size_t i=0;i<N;++i)old[i]=values[i];
    for(std::size_t i=0;i<N;++i)values[i]=old[map[i]];
}
inline void mapInventories(const ModelConfig& saved,const ModelConfig& current,HostState& state) {
    const auto species=order(saved.speciesNames,current.speciesNames,"species");
    const auto condensed=order(saved.condensedNames,current.condensedNames,"condensed species");
    const auto elements=order(saved.elementNames,current.elementNames,"element budget");
    for(auto& q:state.gas)reorder(q.species,species);
    for(auto& q:state.gasChemistryAudit)reorder(q.speciesMassChange,species);
    for(auto& q:state.solid){reorder(q.condensed,condensed);reorder(q.pore,species);}
    for(auto& q:state.film)reorder(q.species,species);
    for(auto* mesh:{&state.gasMesh,&state.solidMesh})
        for(auto& q:mesh->boundaryPrimitive)reorder(q.Y,species);
    for(auto& q:state.ledger){reorder(q.species,species);reorder(q.pore,species);reorder(q.condensed,condensed);}
    reorder(state.budget.boundarySpecies,species);reorder(state.budget.boundaryElements,elements);
    for(auto& q:state.budget.exchangeSpecies)reorder(q,species);
}
inline void requireParticleHistory(const HostState& state,const ModelConfig& config) {
    for(const auto& p:state.particles)if(p.state) {
        const auto& h=p.thermal;
        const auto missing=[&](const char* field){throw std::runtime_error(
            "particle "+std::to_string(p.id)+" requires "+field+" for selected contact model "+std::to_string(config.particleContact.model));};
        if(!h.initialized)missing("initialized thermal/contact history");
        if(!std::isfinite(h.temperature)||!(h.temperature>0))missing("finite positive temperature");
        if(config.particleContact.model==3) {
            for(auto v:h.coldNodeSpecificEnthalpy)
                if(!std::isfinite(v)||!(ugkwp::ParticleCaloric{}.temperature(v)>0))missing("cold1D node enthalpy with positive temperature");
        } else if(config.particleContact.model==4) {
            for(auto v:h.cold2DNodeSpecificEnthalpy)
                if(!std::isfinite(v)||!(ugkwp::ParticleCaloric{}.temperature(v)>0))missing("cold2D node enthalpy with positive temperature");
        }
    }
}
}}
