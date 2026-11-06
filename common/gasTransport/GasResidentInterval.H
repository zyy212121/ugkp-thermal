#pragma once
#include <type_traits>
#include <utility>
#if defined(__CUDACC__)
#define UGKP_INTERVAL_HD __host__ __device__
#else
#define UGKP_INTERVAL_HD
#endif
namespace ugkwp {
// Numerical state remains resident. Only the interval's scalar launch metadata
// changes between graph replays; no solution or matrix is returned to the host.
struct GasResidentInterval {
    double dt=0,begin=0,end=0,middle=0;
    bool active=false,frozen=false;
};
template<class State,class=void>struct HasGasResidentInterval:std::false_type{};
template<class State>struct HasGasResidentInterval<State,std::void_t<
    decltype(std::declval<const State&>().gasResidentInterval)>>:std::true_type{};
template<class State,class Time>
UGKP_INTERVAL_HD inline Time gasResidentStepDt(const State& state,Time value)
{
    if constexpr(HasGasResidentInterval<State>::value)
        if(state.gasResidentInterval.active)return value*Time(state.gasResidentInterval.dt);
    return value;
}
template<class State,class Time>
UGKP_INTERVAL_HD inline Time gasResidentStageTime(const State& state,Time value)
{
    if constexpr(HasGasResidentInterval<State>::value){
        const auto& interval=state.gasResidentInterval;
        if(interval.active){
            if(interval.frozen || value==Time(0))return Time(interval.begin);
            if(value==Time(1))return Time(interval.end);
            if(value==Time(0.5))return Time(interval.middle);
            return Time(interval.begin)+value*Time(interval.dt);
        }
    }
    return value;
}
} // namespace ugkwp
#undef UGKP_INTERVAL_HD
