#pragma once
#include <type_traits>
#include <utility>
#include <cstring>
#if defined(__CUDACC__)
#define UGKP_INTERVAL_HD __host__ __device__
#else
#define UGKP_INTERVAL_HD
#endif
namespace ugkwp {
// Physical interval state is resident. Continuous equal-dt intervals advance
// on device; discontinuous intervals receive exact scalar launch metadata.
struct GasResidentInterval {
    double dt=0,begin=0,end=0,middle=0;
    bool active=false,frozen=false;
};
// Compare launch metadata by value representation, including signed zero.
// Reusing identical metadata must not perform another host-to-device update.
inline bool sameGasResidentInterval(const GasResidentInterval& a,const GasResidentInterval& b){
    return std::memcmp(&a.dt,&b.dt,sizeof(double))==0
        &&std::memcmp(&a.begin,&b.begin,sizeof(double))==0
        &&std::memcmp(&a.end,&b.end,sizeof(double))==0
        &&std::memcmp(&a.middle,&b.middle,sizeof(double))==0
        &&a.active==b.active&&a.frozen==b.frozen;
}
// Continue the physical clock on device only when it reproduces the supplied
// interval exactly. A changed dt, discontinuous clock, or frozen boundary time
// retains the absolute-interval path; no tolerance changes the boundary clock.
inline bool canAdvanceGasResidentInterval(const GasResidentInterval& previous,
    const GasResidentInterval& next){
    if(!previous.active||!next.active||previous.frozen||next.frozen)return false;
    GasResidentInterval advanced=previous;
    advanced.begin=previous.end;
    advanced.end=advanced.begin+previous.dt;
    advanced.middle=advanced.begin+previous.dt*0.5;
    return sameGasResidentInterval(advanced,next);
}
#if defined(__CUDACC__)
template<class State>
__global__ void setResidentGasIntervalKernel(State* state,ugkwp::GasResidentInterval* clock,
    ugkwp::GasResidentInterval interval,bool advance)
{
    if(!blockIdx.x&&!threadIdx.x){
        if(advance){
            interval.begin=clock->end;
            interval.end=__dadd_rn(interval.begin,interval.dt);
            interval.middle=__dadd_rn(interval.begin,__dmul_rn(interval.dt,0.5));
        }
        *clock=interval;state->gasResidentInterval=interval;
    }
}
#endif
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
