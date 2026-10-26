#ifndef CHMT_MATERIALS_MATERIALINVENTORYMATH_H
#define CHMT_MATERIALS_MATERIALINVENTORYMATH_H
#include "coupling/IntervalContracts.H"
#include "../../../common/gpu/ExtendedAccumulator.H"
namespace chmt {
enum class MaterialInventoryCode : int {Success=0,DonorInventory,NonfiniteInventory,NonfiniteEnergy};
// A CSR row contains every solid-consumed packet for this cell, in the original
// packet order (including zero packets). Progress and porosity are unchanged.
CHMT_HD inline MaterialInventoryCode applyMaterialPacketRow(const SolidQ& initial,
    const ExchangePacket* packets,const int* indices,int begin,int end,SolidQ& output){
    using Wide=ugkp::numerics::ExtendedAccumulator;
    Wide sum[Nc+Ns+1],scale[Nc+Ns];
    for(int c=0;c<Nc;++c){sum[c]=Wide(initial.condensed[c]);scale[c]=Wide(absValue(initial.condensed[c]));}
    for(int s=0;s<Ns;++s){sum[Nc+s]=Wide(initial.pore[s]);scale[Nc+s]=Wide(absValue(initial.pore[s]));}
    sum[Nc+Ns]=Wide(initial.energy);
    for(int e=begin;e<end;++e){
        const auto delta=packetDelta(packets[indices[e]]);
        sum[Nc+Ns].add(delta.solid.energy);
        for(int c=0;c<Nc;++c){sum[c].add(delta.solid.condensed[c]);scale[c].add(absValue(delta.solid.condensed[c]));}
        for(int s=0;s<Ns;++s){sum[Nc+s].add(delta.solid.pore[s]);scale[Nc+s].add(absValue(delta.solid.pore[s]));}
    }
    SolidQ result=initial;
    for(int component=0;component<Nc+Ns;++component){
        Real value=sum[component].value();
        if(value<0){
            const auto bound=scale[component].multiplied((std::uint64_t(end-begin)+2)*std::numeric_limits<Real>::epsilon());
            if(sum[component].compareMagnitude(bound)>0)return MaterialInventoryCode::DonorInventory;
            value=0;
        }
        if(!finite(value))return MaterialInventoryCode::NonfiniteInventory;
        if(component<Nc)result.condensed[component]=value;else result.pore[component-Nc]=value;
    }
    result.energy=sum[Nc+Ns].value();
    if(!finite(result.energy))return MaterialInventoryCode::NonfiniteEnergy;
    output=result;return MaterialInventoryCode::Success;
}
inline const char* materialInventoryMessage(MaterialInventoryCode code){
    switch(code){
    case MaterialInventoryCode::DonorInventory:return "material packet batch exceeds physical donor inventory";
    case MaterialInventoryCode::NonfiniteInventory:return "nonfinite material packet result";
    case MaterialInventoryCode::NonfiniteEnergy:return "nonfinite material packet energy";
    default:return "";
    }
}
}
#endif
