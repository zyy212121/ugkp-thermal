#pragma once
#include <algorithm>
namespace ugkwp {
// Match the resident CHT schedule: grow only with a fresh Courant estimate.
// Holding the estimate for N steps must not compound the growth factor N times.
template<class Real> inline Real selectCourantTimeStep(
    Real previous,Real stable,Real growth,Real maximum,bool refreshed) {
    return std::min(maximum,refreshed?std::min(stable,previous*growth):previous);
}
template<class Real> struct GasWindowStep { Real accepted,next; };
// An exact coupling-window tail is integrated without reducing the physical
// timestep seed or causing another growth event at the next window.
template<class Real> inline GasWindowStep<Real> selectGasWindowStep(
    Real requested,Real remaining,Real stable,Real growth,Real maximum,bool refreshed) {
    const Real proposal=selectCourantTimeStep(requested,stable,growth,maximum,refreshed);
    return {std::min(proposal,remaining),proposal};
}
}
