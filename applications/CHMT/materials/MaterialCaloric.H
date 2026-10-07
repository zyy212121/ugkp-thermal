#ifndef CHMT_MATERIALS_MATERIALCALORIC_H
#define CHMT_MATERIALS_MATERIALCALORIC_H
#include "materials/Reaction.H"
namespace chmt {
// Cell-integral caloric polynomial. Formation energy is retained in a0;
// differences/heat capacities never subtract two large formation inventories.
struct MaterialCaloric {
    Real a0=0,a1=0,a2=0,minimumTemperature=0,maximumTemperature=0;
    Real energy(Real T) const { return caloricIntegral(a0,a1,a2,T); }
    Real secant(Real a,Real b) const { return a1+Real(.5)*a2*(a+b); }
    Real capacity(Real T) const { return a1+a2*T; }
};
inline bool materialCaloric(const SolidQ& q,const PhysicsConfig& p,
    MaterialCaloric& out,std::string& error) {
    MaterialCaloric c; c.maximumTemperature=std::numeric_limits<Real>::max(); Real mass=0;
    for(int i=0;i<Nc;++i){
        const Real m=q.condensed[i];const auto& t=p.condensed[i];
        if(!finite(m)||m<0){error="negative/nonfinite condensed caloric inventory";return false;}
        if(m==0)continue;
        if(!validCondensed(t)){error="invalid active condensed caloric table";return false;}
        c.a0+=m*t.e0;c.a1+=m*t.cp0;c.a2+=m*t.cp1;mass+=m;
        c.minimumTemperature=maxValue(c.minimumTemperature,t.Tmin);c.maximumTemperature=minValue(c.maximumTemperature,t.Tmax);
    }
    for(int i=0;i<Ns;++i){
        const Real m=q.pore[i];const auto& t=p.species[i];
        if(!finite(m)||m<0){error="negative/nonfinite pore caloric inventory";return false;}
        if(m==0)continue;
        if(!validSpecies(t)){error="invalid active pore caloric table";return false;}
        c.a0+=m*t.e0;c.a1+=m*(t.cp0-t.R);c.a2+=m*t.cp1;mass+=m;
        c.minimumTemperature=maxValue(c.minimumTemperature,t.Tmin);c.maximumTemperature=minValue(c.maximumTemperature,t.Tmax);
    }
    if(!(mass>0)||!finite(c.a0)||!finite(c.a1)||!finite(c.a2)
        ||!(c.maximumTemperature>c.minimumTemperature)||!(c.capacity(c.minimumTemperature)>0)
        ||!(c.capacity(c.maximumTemperature)>0)) {error="inadmissible material caloric polynomial";return false;}
    out=c;error.clear();return true;
}
}
#endif
