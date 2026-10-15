#ifndef CHMT_GPU_SINGLEGASCONTRACT_H
#define CHMT_GPU_SINGLEGASCONTRACT_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
inline bool validateSingleGasInventory(const ModelConfig& model,const HostState& state,std::string& error){
    const auto& p=model.physics;if(p.gasMode!=ugkwp::GasMode::SingleLegacy)return true;
    const int active=p.singleGasSpecies;
    if(active<0||active>=Ns){error="invalid single gas active species";return false;}
    auto fail=[&](){error="single gas cannot contain or emit another material gas species";return false;};
    for(int s=0;s<Ns;++s)if(s!=active){
        for(const auto& q:state.gas)if(q.species[s]!=0)return fail();
        for(const auto& q:state.solid)if(q.pore[s]!=0)return fail();
        for(const auto& q:state.film)if(q.species[s]!=0)return fail();
        if(p.enableReactions)for(int r=0;r<p.nReactions;++r)if(p.reactions[r].gasNu[s]!=0)return fail();
        if(p.material.enableSurfaceReactions)for(int r=0;r<p.material.nSurfaceReactions;++r)
            if(p.material.surfaceReactions[r].gasNu[s]!=0||p.material.surfaceGasOrder[r][s]!=0)return fail();
        if(p.material.enableMelting&&p.material.phaseFilmY[s]!=0)return fail();
        if(p.material.evaporation!=EvaporationLaw::Disabled&&p.material.evaporationAccommodation[s]!=0)return fail();
        for(std::size_t f=0;f<state.gasMesh.boundaryKind.size();++f)
            if(state.gasMesh.boundaryKind[f]==BoundaryKind::Inlet&&state.gasMesh.boundaryPrimitive[f].Y[s]!=0)return fail();
    }
    error.clear();return true;
}

}
#endif
