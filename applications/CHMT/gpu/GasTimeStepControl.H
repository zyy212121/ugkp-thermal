#pragma once
#include "../../../common/gasNumerics/GasWindowStep.H"
#include <algorithm>
namespace chmt {
// Time-step policy only: the gas update and window geometry are unchanged.
template<class Real> inline Real requestedGasMicrostep(bool adaptive,Real fixedDt,
    Real seed,Real cap,Real maximum) {
    if(!adaptive)return fixedDt;
    const Real limited=std::min(cap,maximum);
    return seed>Real(0)?std::min(limited,seed):limited;
}
template<class Real> inline ugkwp::GasWindowStep<Real> selectGasMicrostep(
    bool adaptive,Real requested,Real remaining,Real stable,Real growth,
    Real maximum,bool refreshed) {
    if(!adaptive)return {std::min(requested,remaining),requested};
    return ugkwp::selectGasWindowStep(requested,remaining,stable,growth,maximum,refreshed);
}
}
