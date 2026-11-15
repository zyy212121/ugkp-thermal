#pragma once
#include "AnchoredParticleRadiationResponse.H"
#include <cmath>
#include <limits>
#if defined(__CUDACC__)
#define UGKP_RAD_DERIV_HD __host__ __device__
#else
#define UGKP_RAD_DERIV_HD
#endif
namespace ugkp { namespace gpu {
template<class Real> struct AnchoredTemperatureDerivative {
    Real derivative=0,capacityMagnitude=0,mass=0;
    bool valid=false,available=false;
};
// One-sided zero-Q derivative of the continuous entrance-enthalpy response.
// Actual rounded publication is still evaluated by the response, not this tangent.
template<class Real,class Caloric>
UGKP_RAD_DERIV_HD AnchoredTemperatureDerivative<Real> anchoredTemperatureDerivative
(const AnchoredParticleRadiationResponse<Real,Caloric>& response,bool heating){
    AnchoredTemperatureDerivative<Real> r;const auto d=response.domain();r.mass=d.mass;
    r.capacityMagnitude=heating?d.upper:-d.lower;r.available=r.capacityMagnitude>Real(0)&&r.mass>Real(0);
    if(!r.available)return r;
    Real sum=0;r.valid=true;
    for(int i=0;i<response.count;++i){const auto& p=response.entries[i];if(!p.eligible||p.mass==Real(0))continue;
        const Real a=std::abs(response.specificCapacity(p,heating));
        const Real t=Real(ugkwp::effectiveParticleTemperature(p.thermal,response.caloric));
        // Resolve phase/table junction to the requested side with the production H' law.
        const Real side=std::nextafter(t,heating?Real(INFINITY):-Real(INFINITY));
        const Real cp=response.caloric.enthalpyDerivative(side);
        if(!std::isfinite(cp)||!(cp>Real(0))){r.valid=false;continue;}
        sum+=p.mass*a/cp;
    }
    r.derivative=sum/(r.mass*r.capacityMagnitude);
    r.valid=r.valid&&std::isfinite(r.derivative);return r;
}
template<class Real> struct BandEmissionDerivative {
    Real band=0,bandDerivative=0,B=0,derivative=0;
    bool valid=false;
};
// Same linear band interpolation / constant extrapolation as mieBand.
// At a knot select the requested side; physical T^4 is never table-clamped.
template<class Real>
UGKP_RAD_DERIV_HD BandEmissionDerivative<Real> bandEmissionDerivative
(const Real* temperature,const Real* band,int count,Real t,bool heating){
    BandEmissionDerivative<Real> r;
    if(count<1||!temperature||!band||!std::isfinite(t))return r;
    int lo=0;
    if(count==1||t<temperature[0]||(t==temperature[0]&&!heating))r.band=band[0];
    else if(t>temperature[count-1]||(t==temperature[count-1]&&heating))r.band=band[count-1];
    else {
        int hi=count-1;
        while(lo+1<hi){const int mid=lo+(hi-lo)/2;
            if(t>temperature[mid]||(t==temperature[mid]&&heating))lo=mid;else hi=mid;}
        const Real width=temperature[lo+1]-temperature[lo];
        if(!(width>Real(0)))return r;
        r.bandDerivative=(band[lo+1]-band[lo])/width;
        r.band=band[lo]+(t-temperature[lo])*r.bandDerivative;
    }
    const Real sigma=Real(5.670374419e-8),t2=t*t,t3=t2*t,t4=t2*t2;
    r.B=Real(4)*sigma*r.band*t4;
    r.derivative=Real(4)*sigma*(r.bandDerivative*t4+Real(4)*r.band*t3);
    r.valid=std::isfinite(r.band)&&std::isfinite(r.bandDerivative)&&std::isfinite(r.B)&&std::isfinite(r.derivative);
    return r;
}
}}
#undef UGKP_RAD_DERIV_HD
