#pragma once
#include "../GpuPrecisionTypes.H"
#include "../GpuParticleCaloric.H"
#include "RadiationImplicitMath.H"
#if defined(__CUDACC__)
#define UGKP_RAD_RESPONSE_HD __host__ __device__
#else
#define UGKP_RAD_RESPONSE_HD
#endif
namespace ugkp { namespace gpu {
// Borrowed resident arrays. A deviceView resolves the owner's current arrays
// after buffer swaps, without publishing their pointer parity to the host.
// Only active, freely moving particles participate, using their ordinary caloric law.
struct ParticleRadiationResponseView {
    bool enabled=false;int capacity=0;
    const int *count=nullptr,*cell=nullptr,*status=nullptr,*cellOffsets=nullptr;
    const unsigned char* wallState=nullptr;
    const GpuReal *mass=nullptr,*temperature=nullptr;
    const double* enthalpyCarry=nullptr;
    double minimumTemperature=0,maximumTemperature=0;
    const ParticleRadiationResponseView* deviceView=nullptr;
};
UGKP_RAD_RESPONSE_HD inline ParticleRadiationResponseView resolveParticleRadiationResponse(ParticleRadiationResponseView view){
    return view.deviceView?*view.deviceView:view;
}
struct CaloricRadiationSample {double temperature=0,energy=0,mass=0;};
UGKP_RAD_RESPONSE_HD inline double radiationSpecificCapacity(double base,double lower,double upper,bool heating,double carry=0.){
    // Subtract the large enthalpies before applying the sub-ULP carry.
    return heating?radiationMath::radiationMaximum(0.,(upper-base)-carry)
        :-radiationMath::radiationMaximum(0.,(base-lower)+carry);
}
// The fraction is the same fraction of each parcel's available specific
// enthalpy used by the resident publication. Heterogeneous initial temperatures
// and sub-temperature-ULP enthalpy carries are retained, never homogenised.
struct IndexedParticleRadiationResponse {
    ParticleRadiationResponseView view;const int* indices;int begin,end;
    UGKP_RAD_RESPONSE_HD CaloricRadiationSample operator()(double fraction,bool heating)const {
        const auto current=resolveParticleRadiationResponse(view);
        const ugkwp::ParticleCaloric law{};CaloricRadiationSample result;
        const double lower=law.enthalpy(current.minimumTemperature),upper=law.enthalpy(current.maximumTemperature);
        for(int slot=begin;slot<end;++slot){const int i=indices?indices[slot]:slot;
            if(current.status[i]!=1||current.wallState[i]!=0)continue;
            const double m=current.mass[i],base=law.enthalpy(double(current.temperature[i])),carry=current.enthalpyCarry[i];
            const double q=fraction*radiationSpecificCapacity(base,lower,upper,heating,carry);
            result.mass+=m;result.temperature+=m*law.temperature(base+(carry+q));result.energy+=m*q;
        }
        if(result.mass>0)result.temperature/=result.mass;
        return result;
    }
};
struct CaloricRadiationResult {double temperature=0,energy=0,residual=0,fraction=0;int iterations=0;};
template<class Response,class Power>
UGKP_RAD_RESPONSE_HD CaloricRadiationResult solveCaloricParticleRadiation(
    const Response& response,const Power& power,double elapsed)
{
    const auto old=response(0.,true);
    if(!(old.mass>0)||elapsed==0)return {old.temperature,0,0,0,0};
    const double initialPower=power(old.temperature);
    if(initialPower==0)return {old.temperature,0,0,0,0};
    const bool heating=initialPower>0;const double sign=heating?1.:-1.;
    const auto bound=response(1.,heating);const double capacity=sign*bound.energy;
    if(!(capacity>0))return {old.temperature,0,-elapsed*initialPower,0,0};
    const auto fractionPower=[&](double fraction){return sign*power(response(fraction,heating).temperature);};
    // Retain the existing endpoint choice when the full interval does not
    // bracket a root. Multi-root/temperature-limit policy is a separate issue.
    double lo=0.,hi=1.,rlo=-elapsed*sign*initialPower;
    const double rhi=capacity-elapsed*fractionPower(hi);
    double fraction=0.;int iterations=0;
    if(!radiationMath::oppositeSigns(rlo,rhi))
        fraction=std::abs(rlo)<=std::abs(rhi)?lo:hi;
    else {
        // An exchange-sized first trial keeps arbitrarily weak, representable
        // heat away from an error floor proportional to the full capacity.
        fraction=radiationMath::radiationMinimum(1.,elapsed*sign*initialPower/capacity);
        for(iterations=1;iterations<=256;++iterations){
            const double exchange=capacity*fraction,source=elapsed*fractionPower(fraction);
            const double residual=exchange-source;
            const double scale=radiationMath::radiationMaximum(std::abs(exchange),std::abs(source));
            if(residual==0.||std::abs(residual)<=64* radiationMath::radiationDoubleEpsilon*scale)break;
            if(radiationMath::oppositeSigns(rlo,residual))hi=fraction;
            else {lo=fraction;rlo=residual;}
            const double next=lo+(hi-lo)*.5;
            if(next==lo||next==hi)break; // representable fraction interval exhausted
            fraction=next;
        }
    }
    const auto accepted=response(fraction,heating);
    return {accepted.temperature,accepted.energy,accepted.energy-elapsed*power(accepted.temperature),fraction,iterations};
}
}}
#undef UGKP_RAD_RESPONSE_HD
