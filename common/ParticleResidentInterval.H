#pragma once
#include <type_traits>
#include <utility>
#if defined(__CUDACC__)
#define UGKP_PARTICLE_INTERVAL_HD __host__ __device__
#else
#define UGKP_PARTICLE_INTERVAL_HD
#endif
namespace ugkwp {
// Particle end time is the caller's committed time, not the gas RK clock.
struct ParticleResidentInterval { double dt=0,end=0;bool active=false; };
template<class State,class=void>struct HasParticleResidentInterval:std::false_type{};
template<class State>struct HasParticleResidentInterval<State,std::void_t<
    decltype(std::declval<const State&>().particleResidentInterval)>>:std::true_type{};
template<class State,class Time>
UGKP_PARTICLE_INTERVAL_HD inline Time particleResidentStepDt(const State& state,Time value){
    if constexpr(HasParticleResidentInterval<State>::value)
        if(state.particleResidentInterval.active)return value*Time(state.particleResidentInterval.dt);
    return value;
}
template<class State,class Time>
UGKP_PARTICLE_INTERVAL_HD inline Time particleResidentEndTime(const State& state,Time value){
    if constexpr(HasParticleResidentInterval<State>::value)
        if(state.particleResidentInterval.active)return Time(state.particleResidentInterval.end);
    return value;
}
}
#undef UGKP_PARTICLE_INTERVAL_HD
