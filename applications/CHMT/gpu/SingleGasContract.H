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
inline bool validateSingleGasProgram(const ModelConfig& model,const WallProgram& program,std::string& error){
    if(model.physics.gasMode!=ugkwp::GasMode::SingleLegacy)return true;
    for(const auto& knot:program.knots)for(const auto& face:knot.faces)
        for(int s=0;s<Ns;++s)if(s!=model.physics.singleGasSpecies
            &&(face.speciesRate[s]!=0||face.poreRate[s]!=0||face.poreSweepRate[s]!=0)){
                error="single gas wall program emits another species";return false;}
    error.clear();return true;
}
}
#endif
