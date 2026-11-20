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
    void add(Number value){
        if constexpr(std::is_same_v<Storage,Number>)sum_+=value;
        else sum_.add(value);
    }
    Number value()const{
        if constexpr(std::is_same_v<Storage,Number>)return sum_;
        else return Number(sum_.value());
    }
};

struct MaterialWindowExchange {
    struct Cell {
        std::array<MaterialWindowSum<PhysicalReal>,Nc+Ns> mass{};
        MaterialWindowSum<EnergyReal> energy;
    };
    std::vector<Cell> cells;
    std::vector<std::array<MaterialWindowSum<PhysicalReal>,Nc>> removed;
    std::uint64_t packets=0;

    explicit MaterialWindowExchange(const HostState& base)
        :cells(base.solid.size()),removed(base.surface.solidFace.size()) {}

    bool append(std::size_t face,const ExchangePacket& packet,const SurfaceMesh& surface,std::string& error) {
        const unsigned consumers=requiredConsumers(packet.kind);
        if((consumers&ConsumeSolid)==0)return true;
        const int cell=packet.solidCell;
        if(face>=removed.size()||face>=surface.solidCell.size()||cell<0||std::size_t(cell)>=cells.size()||cell!=surface.solidCell[face]){
            error="accepted exchange has no matching material owner";return false;
        }
        auto& sum=cells[cell];
        const auto delta=packetDelta(packet).solid;
        if(!finite(delta.energy)){error="nonfinite accepted material energy";return false;}
        sum.energy.add(delta.energy);
        for(int c=0;c<Nc;++c){
            if(!finite(delta.condensed[c])){error="nonfinite accepted condensed mass";return false;}
            sum.mass[c].add(delta.condensed[c]);
            if(packet.kind==ExchangeKind::GasSolid)removed[face][c].add(packet.condensed[c]);
        }
        for(int s=0;s<Ns;++s){
            if(!finite(delta.pore[s])){error="nonfinite accepted pore mass";return false;}
            sum.mass[Nc+s].add(delta.pore[s]);
        }
        ++packets;
        return true;
    }

    bool radiation(std::size_t face,EnergyReal energy,unsigned receiver,const SurfaceMesh& surface,std::string& error){
        if(!finite(energy)){error="nonfinite accepted radiative energy";return false;}
        if(energy==0)return true;
        if(receiver!=ConsumeSolid||face>=surface.solidCell.size()
            ||surface.solidCell[face]<0||std::size_t(surface.solidCell[face])>=cells.size()){
            error="accepted radiation has no material owner";return false;
        }
        cells[surface.solidCell[face]].energy.add(energy);
        return true;
    }
};
}
