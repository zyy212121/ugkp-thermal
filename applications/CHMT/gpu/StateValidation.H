#ifndef CHMT_GPU_STATEVALIDATION_H
#define CHMT_GPU_STATEVALIDATION_H
#include "core/HostState.H"
#include "mesh/Geometry.H"
namespace chmt {
inline bool validateRuntimeState(const ModelConfig& model,HostState& h,std::string& error){
    const auto& p=model.physics;
    if(!h.gas.empty()&&!rebuildGeometry(h.gasMesh,error)&&!detail::completeGeometry(h.gasMesh,error))return false;
    if(!h.solid.empty()&&!rebuildGeometry(h.solidMesh,error)&&!detail::completeGeometry(h.solidMesh,error))return false;
    const bool normal=p.filmThermalMode==FilmThermalMode::ResolvedNormal;
    const std::size_t nf=h.surface.area.size(),ne=h.surface.edgeOwner.size();
    if(!normal&&nf){
        if(!p.enableFilm&&h.filmAux.empty()){
            h.filmAux.resize(nf);for(std::size_t f=0;f<nf;++f){h.filmAux[f].area=h.surface.area[f];h.filmAux[f].normal=h.surface.normal[f];h.filmAux[f].temperature=undefinedValue();}
        }
        if(h.surface.oldArea.empty())h.surface.oldArea=h.surface.area;
        if(h.surface.sweptEdgeArea.empty())h.surface.sweptEdgeArea.assign(ne,0);
        if(h.surface.meshVelocity.empty())h.surface.meshVelocity.assign(nf,Vec3{});
    }
    error.clear();return true;
}
}
#endif
