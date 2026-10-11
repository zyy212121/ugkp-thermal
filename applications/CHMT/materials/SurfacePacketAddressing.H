#ifndef CHMT_MATERIALS_SURFACEPACKETADDRESSING_H
#define CHMT_MATERIALS_SURFACEPACKETADDRESSING_H
#include "core/HostState.H"
#include <unordered_map>
namespace chmt {
// Candidate-local addressing only: geometry metrics may move, but persistent
// surface IDs and their mesh/donor addresses cannot change during this lifetime.
class SurfacePacketAddressing {
    struct Address {std::size_t index;int gasFace,solidCell;};
    std::unordered_map<std::uint64_t,Address> byId_;
    std::uint64_t gasTopology_=0,solidTopology_=0;
    static bool validLayout(const HostState& state,std::size_t n) {
        return state.surface.area.size()==n&&state.surface.persistentId.size()==n
            &&state.surface.gasFace.size()==n&&state.surface.solidCell.size()==n;
    }
public:
    bool build(const HostState& base,const HostState& endpoint,std::string& error) {
        byId_.clear();const std::size_t n=base.surface.area.size();
        if(!validLayout(base,n)||!validLayout(endpoint,n)) {
            error="surface recession identity layout mismatch";return false;
        }
        gasTopology_=base.gasMesh.topologyHash;solidTopology_=base.solidMesh.topologyHash;
        if(endpoint.gasMesh.topologyHash!=gasTopology_||endpoint.solidMesh.topologyHash!=solidTopology_) {
            error="surface recession topology changed within candidate";return false;
        }
        byId_.reserve(n);
        for(std::size_t f=0;f<n;++f) {
            const auto id=base.surface.persistentId[f];
            const int gasFace=base.surface.gasFace[f],solidCell=base.surface.solidCell[f];
            if(endpoint.surface.persistentId[f]!=id||endpoint.surface.gasFace[f]!=gasFace
                ||endpoint.surface.solidCell[f]!=solidCell) {
                error="surface recession addressing changed within candidate";return false;
            }
            if(!byId_.emplace(id,Address{f,gasFace,solidCell}).second) {
                error="ambiguous surface recession packet identity";return false;
            }
        }
        return true;
    }
    bool gasSolidIndex(const HostState& state,const ExchangePacket& packet,
        std::size_t& index,std::string& error) const {
        if(!validLayout(state,byId_.size())||state.gasMesh.topologyHash!=gasTopology_
            ||state.solidMesh.topologyHash!=solidTopology_) {
            error="surface recession identity layout/topology changed";return false;
        }
        const auto found=byId_.find(packet.face);
        if(found==byId_.end()) {error="surface recession packet identity missing";return false;}
        const auto& address=found->second;index=address.index;
        if(state.surface.persistentId[index]!=packet.face||state.surface.gasFace[index]!=address.gasFace
            ||state.surface.solidCell[index]!=address.solidCell) {
            error="surface recession addressing changed within candidate";return false;
        }
        if(address.gasFace<0) {error="surface recession packet identity inactive";return false;}
        const int donor=packet.solidCell;
        if(donor<0||static_cast<std::size_t>(donor)>=state.solid.size()
            ||static_cast<std::size_t>(donor)>=state.solidMesh.volumes.size()||address.solidCell!=donor) {
            error="surface recession packet donor mismatch";return false;
        }
        return true;
    }
};
} // namespace chmt
#endif
