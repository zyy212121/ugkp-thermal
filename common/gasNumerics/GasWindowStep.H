#pragma once
#include <algorithm>
#if defined(__CUDACC__)
#define UGKP_WINDOW_STEP_HD __host__ __device__
#else
#define UGKP_WINDOW_STEP_HD
#endif
namespace ugkwp {


template<class Real> UGKP_WINDOW_STEP_HD inline Real selectCourantTimeStep(
    Real previous,Real stable,Real growth,Real maximum,bool refreshed) {
    return std::min(maximum,refreshed?std::min(stable,previous*growth):previous);
}
template<class Real> struct GasWindowStep { Real accepted,next; };


template<class Real> UGKP_WINDOW_STEP_HD inline GasWindowStep<Real> selectGasWindowStep(
    Real requested,Real remaining,Real stable,Real growth,Real maximum,bool refreshed) {
    const Real proposal=selectCourantTimeStep(requested,stable,growth,maximum,refreshed);
    return {std::min(proposal,remaining),proposal};
}
}

#undef UGKP_WINDOW_STEP_HD
