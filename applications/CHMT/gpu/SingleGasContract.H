#ifndef CHMT_GPU_SINGLEGASCONTRACT_H
#define CHMT_GPU_SINGLEGASCONTRACT_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
inline bool validateSingleGasInventory(const ModelConfig& model,const HostState& state,std::string& error){
    const auto& p=model.physics;if(p.gasMode!=ugkwp::GasMode::SingleLegacy)return true;
    const int active=p.singleGasSpecies;
    if(active<0||active>=Ns){error="invalid single gas active species";return false;}
    auto fail=[&](const char* source,int species,std::size_t index,Real value){error=std::string("single gas missing multi-species operator: source=")+source+" species="+std::to_string(species)+" entity="+std::to_string(index)+" value="+std::to_string(value);return false;};
    for(int s=0;s<Ns;++s)if(s!=active){
        for(std::size_t i=0;i<state.gas.size();++i)if(state.gas[i].species[s]!=0)return fail("gas",s,i,state.gas[i].species[s]);
        for(std::size_t i=0;i<state.solid.size();++i)if(state.solid[i].pore[s]!=0)return fail("solid",s,i,state.solid[i].pore[s]);
        for(std::size_t i=0;i<state.film.size();++i)if(state.film[i].species[s]!=0)return fail("film",s,i,state.film[i].species[s]);
        if(p.enableReactions)for(int r=0;r<p.nReactions;++r)if(p.reactions[r].gasNu[s]!=0)return fail("reaction",s,r,p.reactions[r].gasNu[s]);
        if(p.material.enableSurfaceReactions)for(int r=0;r<p.material.nSurfaceReactions;++r)
            if(p.material.surfaceReactions[r].gasNu[s]!=0||p.material.surfaceGasOrder[r][s]!=0)return fail("surface reaction",s,r,p.material.surfaceReactions[r].gasNu[s]!=0?p.material.surfaceReactions[r].gasNu[s]:p.material.surfaceGasOrder[r][s]);
        if(p.material.enableMelting&&p.material.phaseFilmY[s]!=0)return fail("phase composition",s,0,p.material.phaseFilmY[s]);
        if(p.material.evaporation!=EvaporationLaw::Disabled&&p.material.evaporationAccommodation[s]!=0)return fail("evaporation",s,0,p.material.evaporationAccommodation[s]);
        for(std::size_t f=0;f<state.gasMesh.boundaryKind.size();++f)
            if(state.gasMesh.boundaryKind[f]==BoundaryKind::Inlet&&state.gasMesh.boundaryPrimitive[f].Y[s]!=0)return fail("inlet",s,f,state.gasMesh.boundaryPrimitive[f].Y[s]);
    }
    error.clear();return true;
}

}
#endif
