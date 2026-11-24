#pragma once
#include "GasResidentInterval.H"
#include "GasBoundaryLayerState.H"
#include "GasStateView.H"
#include <type_traits>
#include <utility>
#if defined(__CUDACC__)
#define UGKP_TRANSPORT_PREPARATION_HD __host__ __device__
#else
#define UGKP_TRANSPORT_PREPARATION_HD
#endif
namespace ugkwp {
struct GasTransportPreparation {
    enum : unsigned { primitive=1,gradient=2,limiter=4,eddy=8,sst=16,sensor=32,sstPrimitive=64 };
    unsigned fields=0;
    double time=0;
};
template<class Host,class=void>struct GasTransportPreparationCapability:std::false_type{};
template<class Host>struct GasTransportPreparationCapability<Host,std::void_t<
    decltype(std::declval<Host>().gasCourantPreparation)>>:std::true_type{};
template<class Host>inline GasTransportPreparation* gasTransportPreparation(const Host& state){
    if constexpr(GasTransportPreparationCapability<Host>::value)return state.gasCourantPreparation;
    return nullptr;
}
template<class State>UGKP_TRANSPORT_PREPARATION_HD inline bool gasTransportPrepared(
    const State& state,const GasTransportPreparation* preparation,unsigned fields){
    return preparation && (preparation->fields&fields)==fields
        && preparation->time==double(gasResidentStageTime(state,0.0));
}
UGKP_TRANSPORT_PREPARATION_HD inline void invalidateGasTransportPreparation(GasTransportPreparation* preparation,unsigned fields){
    if(!preparation)return;
#if defined(__CUDA_ARCH__)
    atomicAnd(&preparation->fields,~fields);
#else
    preparation->fields&=~fields;
#endif
}
template<class Host>inline bool gasTransportPreparationEligible(const Host& state){
    return state.hostTurbulenceModel!=0 && !gasBoundaryLayerEnabled(state)
        && !gasMovingGeometry(state) && state.gasSpecies.mode!=GasMode::MixtureChemistry;
}
}
#undef UGKP_TRANSPORT_PREPARATION_HD
