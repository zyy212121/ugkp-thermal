#ifndef CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#define CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#include "materials/StencilMath.H"
namespace chmt {





CHMT_HD inline Real materialReconstructionLimiter(Real value, Real lower,
    Real upper, Real increment) {
    const Real scale=maxValue(absValue(value),maxValue(absValue(lower),absValue(upper)));
    const Real roundoff=64*std::numeric_limits<Real>::epsilon()*scale;
    if (upper-lower<=roundoff && absValue(increment)<=roundoff) return 1;
    return barthJespersen(value,lower,upper,increment);
}
}
#endif
