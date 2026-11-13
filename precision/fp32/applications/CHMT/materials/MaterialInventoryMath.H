#ifndef CHMT_MATERIALS_MATERIALINVENTORYMATH_H
#define CHMT_MATERIALS_MATERIALINVENTORYMATH_H
#include "coupling/IntervalContracts.H"
namespace chmt {
enum class MaterialInventoryCode : int {Success=0,DonorInventory,NonfiniteInventory,NonfiniteEnergy};
// A CSR row contains every solid-consumed packet for this cell, in the original
// packet order (including zero packets). Progress and porosity are unchanged.
template<class Packets>
CHMT_HD inline MaterialInventoryCode applyMaterialPacketRowView(const SolidQ& initial,
    Packets packets,const int* indices,int begin,int end,SolidQ& output){
    PhysicalReal sum[Nc+Ns]{},scale[Nc+Ns]{};
    EnergyReal energy=EnergyReal(initial.energy);
    for(int c=0;c<Nc;++c){sum[c]=initial.condensed[c];scale[c]=absValue(initial.condensed[c]);}
    for(int s=0;s<Ns;++s){sum[Nc+s]=initial.pore[s];scale[Nc+s]=absValue(initial.pore[s]);}
    for(int e=begin;e<end;++e){
        const auto delta=packetDelta(packets[indices[e]]);
        energy+=EnergyReal(delta.solid.energy);
        for(int c=0;c<Nc;++c){sum[c]+=delta.solid.condensed[c];scale[c]+=absValue(delta.solid.condensed[c]);}
        for(int s=0;s<Ns;++s){sum[Nc+s]+=delta.solid.pore[s];scale[Nc+s]+=absValue(delta.solid.pore[s]);}
    }
    SolidQ result=initial;
    for(int component=0;component<Nc+Ns;++component){
        PhysicalReal value=sum[component];
        if(value<0){
            const PhysicalReal bound=scale[component]*PhysicalReal(std::uint64_t(end-begin)+2)*std::numeric_limits<PhysicalReal>::epsilon();
            if(absValue(sum[component])>bound)return MaterialInventoryCode::DonorInventory;
            value=0;
        }
        if(!finite(value))return MaterialInventoryCode::NonfiniteInventory;
        if(component<Nc)result.condensed[component]=value;else result.pore[component-Nc]=value;
    }
    result.energy=energy;
    if(!finite(result.energy))return MaterialInventoryCode::NonfiniteEnergy;
    output=result;return MaterialInventoryCode::Success;
}
// Preserve the public contiguous-storage entry point. Resident history views
// use the same ordered accumulator and donor bound without copying packets.
CHMT_HD inline MaterialInventoryCode applyMaterialPacketRow(const SolidQ& initial,
    const ExchangePacket* packets,const int* indices,int begin,int end,SolidQ& output){
    return applyMaterialPacketRowView(initial,packets,indices,begin,end,output);
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
