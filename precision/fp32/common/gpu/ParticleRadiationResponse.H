#pragma once
#include "RadiationPrecision.H"
#include "../GpuParticleCaloric.H"
#include "RadiationImplicitMath.H"
#include "ParticleRadiationProfile.H"
#if defined(__CUDACC__)
#define UGKP_RAD_RESPONSE_HD __host__ __device__
#else
#define UGKP_RAD_RESPONSE_HD
#endif
namespace ugkp { namespace gpu {



struct ParticleRadiationResponseView {
    bool enabled=false;int capacity=0;
    const int *count=nullptr,*cell=nullptr,*status=nullptr,*cellOffsets=nullptr;
    const unsigned char* wallState=nullptr;
    const GpuReal *mass=nullptr,*temperature=nullptr;
    RadiationReal minimumTemperature=0,maximumTemperature=0;
    const ParticleRadiationResponseView* deviceView=nullptr;
    ParticleRadiationColdProfiles coldProfiles{};
};
UGKP_RAD_RESPONSE_HD inline ParticleRadiationResponseView resolveParticleRadiationResponse(ParticleRadiationResponseView view){
    return view.deviceView?*view.deviceView:view;
}
struct CaloricRadiationSample {RadiationReal temperature=0,energy=0,mass=0;};
UGKP_RAD_RESPONSE_HD inline RadiationReal radiationSpecificCapacity(RadiationReal base,RadiationReal lower,RadiationReal upper,bool heating){
    return heating?radiationMath::radiationMaximum(RadiationReal(0.),upper-base)
        :-radiationMath::radiationMaximum(RadiationReal(0.),base-lower);
}



struct IndexedParticleRadiationResponse {
    ParticleRadiationResponseView view;const int* indices;int begin,end;
    UGKP_RAD_RESPONSE_HD CaloricRadiationSample operator()(RadiationReal fraction,bool heating)const {
        const auto current=resolveParticleRadiationResponse(view);
        const ugkwp::ParticleCaloric law{};CaloricRadiationSample result;
        const RadiationReal lower=law.enthalpy(current.minimumTemperature),upper=law.enthalpy(current.maximumTemperature);
        for(int slot=begin;slot<end;++slot){const int i=indices?indices[slot]:slot;
            if(!radiationParticleEligible(current.status[i],current.wallState[i]))continue;
            const auto profile=current.coldProfiles.profile(i,current.wallState[i]);
            if(profile.nodes){
                accumulateColdRadiationResponse(current,i,profile,RadiationReal(0),fraction,heating,result);
                continue;
            }
            const RadiationReal m=current.mass[i],base=law.enthalpy(RadiationReal(current.temperature[i]));
            const RadiationReal q=fraction*radiationSpecificCapacity(base,lower,upper,heating);
            result.mass+=m;result.temperature+=m*law.temperature(base+q);result.energy+=m*q;
        }
        if(result.mass>0)result.temperature/=result.mass;
        return result;
    }
};
struct CaloricRadiationResult {RadiationReal temperature=0,energy=0,residual=0,fraction=0;int iterations=0;};
template<class Response,class Power>
UGKP_RAD_RESPONSE_HD CaloricRadiationResult solveCaloricParticleRadiation(
    const Response& response,const Power& power,RadiationReal elapsed,
    int maximumIterations=256,RadiationReal energyRelativeTolerance=RadiationReal(64)*radiationMath::radiationDoubleEpsilon)
{
    const auto old=response(RadiationReal(0.),true);
    if(!(old.mass>0)||elapsed==0)return {old.temperature,0,0,0,0};
    const RadiationReal initialPower=power(old.temperature);
    if(initialPower==0)return {old.temperature,0,0,0,0};
    const bool heating=initialPower>0;const RadiationReal sign=heating?RadiationReal(1.):-RadiationReal(1.);
    const auto bound=response(RadiationReal(1.),heating);const RadiationReal capacity=sign*bound.energy;
    if(!(capacity>0))return {old.temperature,0,-elapsed*initialPower,0,0};
    const auto fractionPower=[&](RadiationReal fraction){return sign*power(response(fraction,heating).temperature);};


    RadiationReal lo=RadiationReal(0.),hi=RadiationReal(1.),rlo=-elapsed*sign*initialPower;
    const RadiationReal rhi=capacity-elapsed*fractionPower(hi);
    RadiationReal fraction=RadiationReal(0.);int iterations=0;
    if(!radiationMath::oppositeSigns(rlo,rhi))
        fraction=std::abs(rlo)<=std::abs(rhi)?lo:hi;
    else {


        fraction=radiationMath::radiationMinimum(RadiationReal(1.),elapsed*sign*initialPower/capacity);
        for(iterations=1;iterations<=maximumIterations;++iterations){
            const RadiationReal exchange=capacity*fraction,source=elapsed*fractionPower(fraction);
            const RadiationReal residual=exchange-source;
            const RadiationReal scale=radiationMath::radiationMaximum(std::abs(exchange),std::abs(source));
            if(residual==RadiationReal(0.)||std::abs(residual)<=energyRelativeTolerance*scale)break;
            if(radiationMath::oppositeSigns(rlo,residual))hi=fraction;
            else {lo=fraction;rlo=residual;}
            const RadiationReal next=lo+(hi-lo)*RadiationReal(.5);
            if(next==lo||next==hi)break;
            fraction=next;
        }
    }
    const auto accepted=response(fraction,heating);
    return {accepted.temperature,accepted.energy,accepted.energy-elapsed*power(accepted.temperature),fraction,iterations};
}
}}
#undef UGKP_RAD_RESPONSE_HD
