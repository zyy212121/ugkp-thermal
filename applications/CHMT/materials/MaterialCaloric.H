#ifndef CHMT_MATERIALS_MATERIALCALORIC_H
#define CHMT_MATERIALS_MATERIALCALORIC_H
#include "materials/Reaction.H"
namespace chmt {
// Condensed caloric polynomial plus shared gas calorics. Stable secants never
// subtract two large formation inventories within a thermo branch.
struct MaterialCaloric {
    Real a0=0,a1=0,a2=0,minimumTemperature=0,maximumTemperature=0;
    SpeciesThermo gas[Ns]{};Real pore[Ns]{};
    template<bool Terminal=false>
    Real energy(Real T) const {
        Real result=caloricIntegral(a0,a1,a2,T);
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesE(gas[s],T);
        return result;
    }
    template<bool Terminal=false>
    Real secant(Real a,Real b) const {
        Real result=a1+Real(.5)*a2*(a+b);
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesEnergySecant<Terminal>(gas[s],a,b);
        return result;
    }
    Real capacity(Real T) const {
        Real result=a1+a2*T;
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesCv(gas[s],T);
        return result;
    }
};
template<bool Terminal=false>
inline bool materialCaloric(const SolidQ& q,const PhysicsConfig& p,
    MaterialCaloric& out,std::string& error,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    MaterialCaloric c; c.maximumTemperature=std::numeric_limits<Real>::max(); Real mass=0;
    for(int i=0;i<Nc;++i){
        const Real m=q.condensed[i];const auto& t=p.condensed[i];
        if((!finite(m)||m<0)){error="negative/nonfinite condensed caloric inventory";return false;}
        if(m==0)continue;
        if(!validCondensed(t)){error="invalid active condensed caloric table";return false;}
        c.a0+=m*t.e0;c.a1+=m*t.cp0;c.a2+=m*t.cp1;mass+=m;
        c.minimumTemperature=maxValue(c.minimumTemperature,t.Tmin);c.maximumTemperature=minValue(c.maximumTemperature,t.Tmax);
    }
    for(int i=0;i<Ns;++i){
        const Real m=q.pore[i];const auto& t=p.species[i];
        if((!finite(m)||m<0)){error="negative/nonfinite pore caloric inventory";return false;}
        if(m==0)continue;
        if(!validSpecies(t)){if(numericalFailure)*numericalFailure=t.model==ugkwp::SpeciesThermoModel::LinearCp||t.model==ugkwp::SpeciesThermoModel::NASA7;error="invalid active pore caloric table";return false;}
        c.gas[i]=t;c.pore[i]=m;mass+=m;
        c.minimumTemperature=maxValue(c.minimumTemperature,t.Tmin);c.maximumTemperature=minValue(c.maximumTemperature,t.Tmax);
    }
    if((!(mass>0)||!finite(c.a0)||!finite(c.a1)||!finite(c.a2))) {error="inadmissible material caloric state";return false;}
    out=c;error.clear();return true;
}
}
#endif
