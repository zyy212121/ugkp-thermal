#pragma once
#include "core/HostState.H"
namespace chmt {
// A gas cell owns one conservative reactant inventory, even if several of
// its faces react. Partition that inventory over the exposed physical areas
// for one explicit surface interval. The partition changes only an active
// donor constraint, not the unconstrained kinetic law. Unused shares are not
// borrowed during this interval; temporal refinement reduces that splitting
// error without changing the global flow step because of reactant starvation.
inline bool surfaceGasDonorShares(const HostState& h,std::vector<Real>& shares,std::string& error) {
    const auto& surface=h.surface;
    if(surface.gasFace.size()!=surface.area.size()){
        error="surface gas donor face layout mismatch";return false;
    }
    std::vector<Real> areas(h.gasMesh.volumes.size(),0);
    shares.assign(surface.area.size(),0);
    for(std::size_t i=0;i<surface.area.size();++i){
        const int face=surface.gasFace[i];if(face<0)continue;
        if(std::size_t(face)>=h.gasMesh.owner.size()||std::size_t(face)>=h.gasMesh.areaVectors.size()){
            error="surface gas donor face outside mesh";return false;
        }
        const int cell=h.gasMesh.owner[face];
        if(cell<0||std::size_t(cell)>=areas.size()){
            error="surface gas donor cell outside mesh";return false;
        }
        shares[i]=mag(h.gasMesh.areaVectors[face]);areas[cell]+=shares[i];
    }
    for(std::size_t i=0;i<shares.size();++i)if(surface.gasFace[i]>=0){
        const Real total=areas[h.gasMesh.owner[surface.gasFace[i]]];
        shares[i]=total>0?shares[i]/total:0;
    }
    return true;
}
}
