#pragma once
#ifdef __CUDACC__
#define UGKP_INJECTION_HD __host__ __device__
#else
#define UGKP_INJECTION_HD
#endif
namespace ugkwp {
template<class Scalar>
UGKP_INJECTION_HD inline Scalar particleInjectionVelocityComponent
(bool specified, double configured, Scalar gasVelocity)
{return specified ? Scalar(configured) : gasVelocity;}
}
#undef UGKP_INJECTION_HD
