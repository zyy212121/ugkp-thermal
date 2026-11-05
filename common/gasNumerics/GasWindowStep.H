#pragma once
#include <algorithm>
namespace ugkwp {
template<class Real> struct GasWindowStep { Real accepted,next; };
// Keep the stability/growth proposal separate from the step clipped to a
// coupling boundary. The exact (possibly sub-ulp) tail is still integrated;
// it is not a new physical stability limit for the following interval.
template<class Real> inline GasWindowStep<Real> selectGasWindowStep(
    Real requested,Real remaining,Real stable,Real growth,Real maximum) {
    const Real proposal=std::min(requested,stable);
    return {std::min(proposal,remaining),std::min(maximum,growth*proposal)};
}
}
