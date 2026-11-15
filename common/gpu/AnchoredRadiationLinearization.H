#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKP_ANCHORED_LINEAR_HD __host__ __device__
#else
#define UGKP_ANCHORED_LINEAR_HD
#endif
namespace ugkp { namespace gpu {



template<class Real> struct AnchoredRadiationLinearization {
    Real b=0,f=0,linearizedEmission=0,thermalFeedback=0,dtA=0,B0=0;
    bool valid=false,nonnegativeSlope=false;
    UGKP_ANCHORED_LINEAR_HD Real predictedQ(Real G)const{return dtA*f*(G-B0);}
};
template<class Real>
UGKP_ANCHORED_LINEAR_HD AnchoredRadiationLinearization<Real> linearizeAnchoredRadiation
(Real Qtrial,Real Btrial,Real B0,Real zeroSideDerivative,Real dt,Real A){
    AnchoredRadiationLinearization<Real> r;
    r.b=Qtrial!=Real(0)?(Btrial-B0)/Qtrial:zeroSideDerivative;
    r.nonnegativeSlope=r.b>=Real(0);r.dtA=dt*A;r.B0=B0;
    const Real denominator=Real(1)+r.dtA*r.b;
    r.valid=std::isfinite(Qtrial)&&std::isfinite(Btrial)&&std::isfinite(B0)
        &&std::isfinite(zeroSideDerivative)&&std::isfinite(dt)&&std::isfinite(A)
        &&std::isfinite(r.b)&&std::isfinite(r.dtA)&&std::isfinite(denominator)&&denominator>Real(0);
    if(!r.valid)return r;
    r.f=Real(1)/denominator;
    const Real fourPi=Real(12.566370614359172953850573533118);
    r.linearizedEmission=r.f*B0/fourPi;
    r.thermalFeedback=Real(1)-r.f;
    r.valid=std::isfinite(r.f)&&std::isfinite(r.linearizedEmission)&&std::isfinite(r.thermalFeedback);
    return r;
}
template<class Real> struct MaterialRadiationResidual {
    Real requestedByPhysics=0,residual=0,scale=0,normalized=0;
    bool valid=false;
};


template<class Real>
UGKP_ANCHORED_LINEAR_HD MaterialRadiationResidual<Real> materialRadiationResidual
(Real actualQ,Real Bactual,Real kappaActual,Real volume,Real G,Real dt,Real scaleFloor){
    MaterialRadiationResidual<Real> r;
    r.requestedByPhysics=dt*kappaActual*volume*(G-Bactual);
    r.residual=actualQ-r.requestedByPhysics;
    const Real aq=std::abs(actualQ),ap=std::abs(r.requestedByPhysics);
    r.scale=aq>ap?aq:ap;if(scaleFloor>r.scale)r.scale=scaleFloor;
    r.normalized=r.scale>Real(0)?std::abs(r.residual)/r.scale:Real(0);
    r.valid=std::isfinite(actualQ)&&std::isfinite(Bactual)&&std::isfinite(kappaActual)
        &&std::isfinite(volume)&&std::isfinite(G)&&std::isfinite(dt)&&std::isfinite(scaleFloor)
        &&scaleFloor>=Real(0)&&std::isfinite(r.requestedByPhysics)&&std::isfinite(r.residual)
        &&std::isfinite(r.scale)&&std::isfinite(r.normalized);
    return r;
}
}}
#undef UGKP_ANCHORED_LINEAR_HD
