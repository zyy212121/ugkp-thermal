#ifndef CHMT_CONFIGURATION_COUPLINGENTRY_H
#define CHMT_CONFIGURATION_COUPLINGENTRY_H
#include "core/HostState.H"
#include <set>
namespace chmt {
// Frontend ownership check. Material unit tests may use isolated participants;
// the CHMT application itself requires an actual gas/material interface.
inline bool validateCouplingEntry(const std::string& mode,const HostState& state,std::string& error) {
    auto fail=[&](const char* message){error=message;return false;};
    if(mode!="Multirate")return fail("CHMT requires executionMode Multirate; pure gas belongs to gasUGKP");
    if(state.gas.empty()||state.gas.size()!=state.gasMesh.volumes.size())
        return fail("CHMT requires populated gas mesh and inventory");
    if(state.solid.empty()||state.solid.size()!=state.solidMesh.volumes.size())
        return fail("CHMT requires populated coupled material mesh and inventory");
    const auto& surface=state.surface;const auto count=surface.area.size();
    if(!count||surface.gasFace.size()!=count||surface.solidFace.size()!=count
        ||surface.solidCell.size()!=count||surface.persistentId.size()!=count)
        return fail("CHMT requires a complete gas/material interface map");
    std::set<int> gasReceivers;std::set<std::uint64_t> identities;
    for(std::size_t f=0;f<count;++f){
        const int g=surface.gasFace[f],s=surface.solidFace[f],c=surface.solidCell[f];
        if(!finite(surface.area[f])||surface.area[f]<=0||g<0||s<0||c<0
            ||std::size_t(g)>=state.gasMesh.owner.size()||std::size_t(g)>=state.gasMesh.neighbour.size()
            ||std::size_t(g)>=state.gasMesh.boundaryKind.size()||std::size_t(s)>=state.solidMesh.owner.size()
            ||std::size_t(s)>=state.solidMesh.neighbour.size()||std::size_t(c)>=state.solid.size()
            ||state.gasMesh.owner[g]<0||std::size_t(state.gasMesh.owner[g])>=state.gas.size()
            ||state.gasMesh.neighbour[g]>=0||state.solidMesh.neighbour[s]>=0
            ||state.solidMesh.owner[s]!=c||state.gasMesh.boundaryKind[g]!=BoundaryKind::Interface)
            return fail("CHMT requires valid boundary receivers on both coupled participants");
        if(!gasReceivers.insert(g).second||!identities.insert(surface.persistentId[f]).second)
            return fail("CHMT interface receivers and persistent identities must be unique");
    }
    error.clear();return true;
}
}
#endif
