#pragma once
#include <type_traits>
#include <utility>
namespace mixtureParticles {
template<class State>
__device__ int activeCount(const State& s)
{return *s.particleCountDevice<0?0:(*s.particleCountDevice>s.particleCapacity?s.particleCapacity:*s.particleCountDevice);}
template<class State,class=void>struct HasContactState:std::false_type{};
template<class State>struct HasContactState<State,std::void_t<
    decltype(std::declval<const State&>().pStuck)>>:std::true_type{};
template<class State>
__device__ bool mobile(const State& s,int i) {
    if constexpr(HasContactState<State>::value)return s.pStuck[i]==0;
    return true;
}
template<class State>
__device__ bool member(const State& s,int i,int c)
{return i>=0 && i<activeCount(s) && s.pStatus[i]==1 && s.pCellId[i]==c && mobile(s,i);}
template<class State,class=void>struct HasSplitDirectory:std::false_type{};
template<class State>struct HasSplitDirectory<State,std::void_t<
    decltype(std::declval<const State&>().preBaseCellOffset)>>:std::true_type{};

template<class State>
__device__ int count(const State&s,int c,bool useCsr,bool useSplit)
{
    if constexpr(HasSplitDirectory<State>::value)
        if(useSplit)return s.preBaseCellOffset[c+1]-s.preBaseCellOffset[c]
            +s.cellParticleOffset[c+1]-s.cellParticleOffset[c];
    return useCsr?s.cellParticleOffset[c+1]-s.cellParticleOffset[c]:activeCount(s);
}
template<class State>
__device__ int parcel(const State&s,int c,int position,bool useCsr,bool useSplit)
{
    if constexpr(HasSplitDirectory<State>::value)
        if(useSplit)
        {
            const int baseCount=s.preBaseCellOffset[c+1]-s.preBaseCellOffset[c];
            return position<baseCount?s.preBaseCellOffset[c]+position
                :s.sortedParticleIndex[s.cellParticleOffset[c]+position-baseCount];
        }
    return useCsr?s.sortedParticleIndex[s.cellParticleOffset[c]+position]:position;
}

}
