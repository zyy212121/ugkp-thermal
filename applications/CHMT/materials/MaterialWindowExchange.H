#pragma once
#include "core/HostState.H"
#include "coupling/ExchangeLedger.H"
#include "../../../common/gpu/ExtendedAccumulator.H"
#include <array>
#include <type_traits>

namespace chmt {
template<class Number>
class MaterialWindowSum {
    using Storage=std::conditional_t<std::is_same_v<Number,float>,Number,ugkp::numerics::ExtendedAccumulator>;
    Storage sum_{};
public:
    CHMT_HD void add(Number value){
        if constexpr(std::is_same_v<Storage,Number>)sum_+=value;
        else sum_.add(value);
    }
    CHMT_HD Number value()const{
        if constexpr(std::is_same_v<Storage,Number>)return sum_;
        else return Number(sum_.value());
    }
};

struct MaterialWindowCell {
    MaterialWindowSum<PhysicalReal> mass[Nc+Ns]{};
    MaterialWindowSum<EnergyReal> energy;
};

CHMT_HD inline const char* addMaterialWindowPacket(MaterialWindowCell& sum,
    MaterialWindowSum<PhysicalReal>* removed,const ExchangePacket& packet,int cell){
    if((requiredConsumers(packet.kind)&ConsumeSolid)==0)return nullptr;
    if(packet.solidCell!=cell)return "accepted exchange has no matching material owner";
    const auto delta=packetDelta(packet).solid;
    if(!finite(delta.energy))return "nonfinite accepted material energy";
    sum.energy.add(delta.energy);
    for(int c=0;c<Nc;++c){
        if(!finite(delta.condensed[c]))return "nonfinite accepted condensed mass";
        sum.mass[c].add(delta.condensed[c]);
        if(packet.kind==ExchangeKind::GasSolid)removed[c].add(packet.condensed[c]);
    }
    for(int s=0;s<Ns;++s){
        if(!finite(delta.pore[s]))return "nonfinite accepted pore mass";
        sum.mass[Nc+s].add(delta.pore[s]);
    }
    return nullptr;
}

CHMT_HD inline const char* addMaterialWindowRadiation(MaterialWindowCell& sum,EnergyReal energy,unsigned receiver){
    if(!finite(energy))return "nonfinite accepted radiative energy";
    if(energy==0)return nullptr;
    if(receiver!=ConsumeSolid)return "accepted radiation has no material owner";
    sum.energy.add(energy);
    return nullptr;
}

struct MaterialWindowExchange {
    using Removal=std::array<MaterialWindowSum<PhysicalReal>,Nc>;
    std::vector<MaterialWindowCell> cells;
    std::vector<Removal> removed;

    explicit MaterialWindowExchange(const HostState& base)
        :cells(base.solid.size()),removed(base.surface.solidFace.size()) {}

    bool append(std::size_t face,const ExchangePacket& packet,const SurfaceMesh& surface,std::string& error) {
        if((requiredConsumers(packet.kind)&ConsumeSolid)==0)return true;
        const int cell=packet.solidCell;
        if(face>=removed.size()||face>=surface.solidCell.size()||cell<0||std::size_t(cell)>=cells.size()||cell!=surface.solidCell[face]){
            error="accepted exchange has no matching material owner";return false;
        }
        if(const auto* failure=addMaterialWindowPacket(cells[cell],removed[face].data(),packet,cell)){
            error=failure;return false;
        }
        return true;
    }

    bool radiation(std::size_t face,EnergyReal energy,unsigned receiver,const SurfaceMesh& surface,std::string& error){
        if(energy==0)return true;
        if(face>=surface.solidCell.size()||surface.solidCell[face]<0||std::size_t(surface.solidCell[face])>=cells.size()){
            error="accepted radiation has no material owner";return false;
        }
        if(const auto* failure=addMaterialWindowRadiation(cells[surface.solidCell[face]],energy,receiver)){
            error=failure;return false;
        }
        return true;
    }
};
}
