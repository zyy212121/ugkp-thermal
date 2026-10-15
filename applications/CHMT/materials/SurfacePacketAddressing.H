#ifndef CHMT_MATERIALS_SURFACEPACKETADDRESSING_H
#define CHMT_MATERIALS_SURFACEPACKETADDRESSING_H
#include "core/HostState.H"
#include <unordered_map>
namespace chmt {
class SurfacePacketAddressing {
    std::unordered_map<std::uint64_t,std::size_t> byId_;
public:
    bool build(const HostState& base,const HostState&,std::string& error) {
        byId_.clear();const std::size_t n=base.surface.area.size();byId_.reserve(n);
        for(std::size_t f=0;f<n;++f)byId_.emplace(base.surface.persistentId[f],f);
        error.clear();return true;
    }
    bool gasSolidIndex(const HostState&,const ExchangePacket& packet,
        std::size_t& index,std::string& error) const {
        const auto found=byId_.find(packet.face);
        if(found==byId_.end()){error="surface recession packet identity missing";return false;}
        index=found->second;error.clear();return true;
    }
};
}
#endif
