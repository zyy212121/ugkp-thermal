#pragma once
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#define UGKWP_PSEUDO_TIME_HD __host__ __device__
#else
#define UGKWP_PSEUDO_TIME_HD
#endif

namespace ugkwp
{
// Optional steady capability: legacy and coupled state layouts need no fields.
template<class State, class = void> struct HasGasPseudoTime : std::false_type {};
template<class State> struct HasGasPseudoTime<State, std::void_t<
    decltype(std::declval<const State&>().gasPseudoDt),
    decltype(std::declval<const State&>().gasPseudoReferenceDt)>> : std::true_type {};

template<class State, class Time>
UGKWP_PSEUDO_TIME_HD inline Time gasCellTimeStep(const State& s, int cell, Time dt)
{
    if constexpr (HasGasPseudoTime<State>::value)
        if (s.gasPseudoDt)
            // Preserve source/subinterval fractions of the reference step.
            return dt * Time(s.gasPseudoDt[cell] / s.gasPseudoReferenceDt);
    return dt;
}

template<class State, class = void> struct HasGasSteadyTime : std::false_type {};
template<class State> struct HasGasSteadyTime<State, std::void_t<
    decltype(std::declval<const State&>().gasSteadyMode),
    decltype(std::declval<const State&>().gasSteadyBoundaryTime)>> : std::true_type {};

// Boundary scheduling is independent of local pseudo-time/source composition.
// Statistical particle sampling freezes forcing while keeping physical steps.
template<class State, class = void> struct HasGasFrozenBoundaryTime : std::false_type {};
template<class State> struct HasGasFrozenBoundaryTime<State, std::void_t<
    decltype(std::declval<const State&>().gasSteadyBoundaryFrozen),
    decltype(std::declval<const State&>().gasSteadyBoundaryTime)>> : std::true_type {};

template<class State, class Time>
UGKWP_PSEUDO_TIME_HD inline Time gasStageTime(const State& s, Time time)
{
    if constexpr (HasGasFrozenBoundaryTime<State>::value)
    {
        if (s.gasSteadyBoundaryFrozen) return Time(s.gasSteadyBoundaryTime);
    }
    else if constexpr (HasGasSteadyTime<State>::value)
        if (s.gasSteadyMode) return Time(s.gasSteadyBoundaryTime);
    return time;
}
}
#undef UGKWP_PSEUDO_TIME_HD
