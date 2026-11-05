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
// Borrowed resident arrays, valid until the owner advances or changes storage.
// Only active, freely moving particles participate, using their ordinary caloric law.
struct ParticleRadiationResponseView {
    bool enabled=false;int capacity=0;
    const int *count=nullptr,*cell=nullptr,*status=nullptr,*cellOffsets=nullptr;
    const unsigned char* wallState=nullptr;
    const GpuReal *mass=nullptr,*temperature=nullptr;
    const double* enthalpyCarry=nullptr;
    double minimumTemperature=0,maximumTemperature=0;
};
struct CaloricRadiationSample {double temperature=0,energy=0,mass=0;};
UGKP_RAD_RESPONSE_HD inline double radiationSpecificCapacity(double h,double lower,double upper,bool heating){
    return heating?radiationMath::radiationMaximum(0.,upper-h):-radiationMath::radiationMaximum(0.,h-lower);
}
// The fraction is the same fraction of each parcel's available specific
// enthalpy used by the resident publication. Heterogeneous initial temperatures
// and sub-temperature-ULP enthalpy carries are retained, never homogenised.
struct IndexedParticleRadiationResponse {
    ParticleRadiationResponseView view;const int* indices;int begin,end;
    UGKP_RAD_RESPONSE_HD CaloricRadiationSample operator()(double fraction,bool heating)const {
        const ugkwp::ParticleCaloric law{};CaloricRadiationSample result;
        const double lower=law.enthalpy(view.minimumTemperature),upper=law.enthalpy(view.maximumTemperature);
        for(int slot=begin;slot<end;++slot){const int i=indices?indices[slot]:slot;
            if(view.status[i]!=1||view.wallState[i]!=0)continue;
            const double m=view.mass[i],h=law.enthalpy(double(view.temperature[i]))+view.enthalpyCarry[i];
            const double q=fraction*radiationSpecificCapacity(h,lower,upper,heating);
            result.mass+=m;result.temperature+=m*law.temperature(h+q);result.energy+=m*q;
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
    const auto root=radiationMath::solveImplicitMaterialTemperature(
        0.,1.,0.,1.,capacity,elapsed,100,1e-12,1e-10,fractionPower);
    const auto accepted=response(root.temperatureK,heating);
    return {accepted.temperature,accepted.energy,accepted.energy-elapsed*power(accepted.temperature),root.temperatureK,root.iterations};
}
}}
#undef UGKP_RAD_RESPONSE_HD
