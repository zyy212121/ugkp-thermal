#pragma once
#include "../../../common/gasNumerics/GasWindowStep.H"
#include <algorithm>
#ifdef __CUDACC__
#define CHMT_STEP_HD __host__ __device__
#else
#define CHMT_STEP_HD
#endif
namespace chmt {

template<class Real> CHMT_STEP_HD inline Real requestedGasMicrostep(bool adaptive,Real fixedDt,
    Real seed,Real cap,Real maximum) {
    if(!adaptive)return fixedDt;
    const Real limited=std::min(cap,maximum);
    return seed>Real(0)?std::min(limited,seed):limited;
}
template<class Real> CHMT_STEP_HD inline ugkwp::GasWindowStep<Real> selectGasMicrostep(
    bool adaptive,Real requested,Real remaining,Real stable,Real growth,
    Real maximum,bool refreshed) {
    if(!adaptive)return {std::min(requested,remaining),requested};
    return ugkwp::selectGasWindowStep(requested,remaining,stable,growth,maximum,refreshed);
}
}

#undef CHMT_STEP_HD
