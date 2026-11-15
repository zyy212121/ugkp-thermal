#pragma once
// Scalar math must not depend on inlet configuration.
template<class Real>
__device__ Real sqr3(const Real x,const Real y,const Real z) { return x*x+y*y+z*z; }
