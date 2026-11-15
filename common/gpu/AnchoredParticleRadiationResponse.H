#pragma once
#include "../GpuParticleThermalCarry.H"
#include <cmath>
#include "MaterialPublicationResolution.H"
#if defined(__CUDACC__)
#define UGKP_ANCHORED_RAD_HD __host__ __device__
#else
#define UGKP_ANCHORED_RAD_HD
#endif
namespace ugkp { namespace gpu {


template<class Real> struct RadiationEntryParticle {
    Real mass=0;
    ugkwp::ParticleThermalCarry<Real> thermal{};
    bool eligible=false;
};
namespace anchoredRadiationDetail {
template<class Real,class State>
UGKP_ANCHORED_RAD_HD auto carry(const State& s,int)->decltype(Real(s.specificEnthalpyResidual)) {return Real(s.specificEnthalpyResidual);}
template<class Real,class State>
UGKP_ANCHORED_RAD_HD Real carry(const State&,long) {return Real(0);}
template<class Real> UGKP_ANCHORED_RAD_HD Real positive(Real x){return x>Real(0)?x:Real(0);}
}
template<class Real> struct RadiationEnergyDomain {
    Real lower=0,upper=0,mass=0,entryTemperature=0;
    Real entryTemperatureGridAllowance=0;
    UGKP_ANCHORED_RAD_HD bool contains(Real q)const{return q>=lower&&q<=upper;}
};
template<class Real> struct AnchoredRadiationSample {
    Real requestedEnergy=0,actualEnergy=0,temperature=0,mass=0,fraction=0;
    Real temperatureGridAllowance=0;
    RadiationEnergyDomain<Real> domain{};
    bool withinDomain=false,publicationValid=true;
};
template<class Real,class Caloric> struct AnchoredParticleRadiationResponse {
    const RadiationEntryParticle<Real>* entries=nullptr;
    int count=0;
    Real minimumTemperature=0,maximumTemperature=0;
    Caloric caloric{};
    UGKP_ANCHORED_RAD_HD Real specificCapacity(const RadiationEntryParticle<Real>& p,bool heating)const {
        const Real base=caloric.enthalpy(p.thermal.temperature);
        const Real carry=anchoredRadiationDetail::carry<Real>(p.thermal,0);

        return heating?anchoredRadiationDetail::positive((caloric.enthalpy(maximumTemperature)-base)-carry)
                      :-anchoredRadiationDetail::positive((base-caloric.enthalpy(minimumTemperature))+carry);
    }
    UGKP_ANCHORED_RAD_HD RadiationEnergyDomain<Real> domain()const {
        RadiationEnergyDomain<Real> d;
        for(int i=0;i<count;++i){const auto& p=entries[i];if(!p.eligible)continue;
            d.entryTemperatureGridAllowance+=p.mass*publicationTemperatureGrid<Real>(p.thermal.temperature,minimumTemperature,maximumTemperature,caloric);
            d.lower+=p.mass*specificCapacity(p,false);d.upper+=p.mass*specificCapacity(p,true);d.mass+=p.mass;d.entryTemperature+=p.mass*Real(ugkwp::effectiveParticleTemperature(p.thermal,caloric));}
        if(d.mass>Real(0))d.entryTemperature/=d.mass;
        return d;
    }
    UGKP_ANCHORED_RAD_HD AnchoredRadiationSample<Real> operator()(Real q)const {return evaluate(q,domain());}
    UGKP_ANCHORED_RAD_HD AnchoredRadiationSample<Real> evaluate(Real q,const RadiationEnergyDomain<Real>& cachedDomain)const {
        AnchoredRadiationSample<Real> out;out.requestedEnergy=q;out.domain=cachedDomain;out.mass=out.domain.mass;



        out.temperatureGridAllowance=out.domain.entryTemperatureGridAllowance;
        out.withinDomain=out.domain.contains(q);

        if(!out.withinDomain)return out;
        const bool heating=q>=Real(0);
        const Real bound=heating?out.domain.upper:out.domain.lower;
        out.fraction=bound!=Real(0)?q/bound:Real(0);
        for(int i=0;i<count;++i){const auto& p=entries[i];if(!p.eligible)continue;
            auto candidate=p.thermal;
            const Real dq=out.fraction*specificCapacity(p,heating);
            const auto published=ugkwp::publishParticleThermalHeat(candidate,dq,minimumTemperature,maximumTemperature,caloric);
            out.publicationValid=out.publicationValid&&published.valid;
            out.actualEnergy+=p.mass*Real(published.acceptedSpecificHeat);
            out.temperatureGridAllowance+=p.mass*publicationTemperatureGrid<Real>(candidate.temperature,minimumTemperature,maximumTemperature,caloric);
            out.temperature+=p.mass*Real(ugkwp::effectiveParticleTemperature(candidate,caloric));
        }
        if(out.mass>Real(0))out.temperature/=out.mass;
        return out;
    }
};
}}
#undef UGKP_ANCHORED_RAD_HD
