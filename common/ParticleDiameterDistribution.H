#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKP_RR_HD __host__ __device__
#else
#define UGKP_RR_HD
#endif
namespace ugkwp {
// Doubly truncated RR / Weibull inverse CDF. Equal-mass parcels sample the
// represented mass distribution; this is not a number-weighted conversion.
// scale is the untruncated 63.212% diameter, n is the positive shape exponent.
template<class Real>
UGKP_RR_HD inline Real rosinRammlerDiameter
(Real scale, Real n, Real lower, Real upper, Real u) {
    if (upper<=lower || u<=Real(0)) return lower;
    if (u>=Real(1)) return upper;
    const Real a=::pow(lower/scale,n);
    const Real b=::pow(upper/scale,n);
    // When a overflows, the conditional tail is narrower than one diameter ulp.
    if (std::isinf(a)) return lower;
    const Real logRatio=n*::log(upper/lower);
    const Real delta=(a>Real(0) && logRatio<Real(1))
        ? a*::expm1(logRatio) : b-a;
    Real d;
    if (delta==Real(0)) {
        // Small-q limit: exp(-q)=1. Avoid a vanishing CDF normalization.
        const Real ratio=::exp(-logRatio);
        d=upper*::pow(ratio+u*(Real(1)-ratio),Real(1)/n);
    } else {
        const Real increment=-::log1p(u*::expm1(-delta));
        d=a>Real(0)
            ? lower*::exp(::log1p(increment/a)/n)
            : scale*::pow(increment,Real(1)/n);
    }
    return d<lower?lower:(d>upper?upper:d);
}
}
#undef UGKP_RR_HD
