#ifndef UGKP_GPU_MIE_OPTICAL_MATH_H
#define UGKP_GPU_MIE_OPTICAL_MATH_H
#include <cmath>
#include <cstddef>
#ifdef __CUDACC__
#define UGKP_MIE_HD __host__ __device__
#else
#define UGKP_MIE_HD
#endif
namespace ugkp { namespace gpu {
struct MieOpticalTableView {
    int temperatures=0,diameters=0,angles=0;
    const double *temperature=nullptr,*diameter=nullptr,*mu=nullptr,*band=nullptr;
    const double *absorption=nullptr,*scattering=nullptr,*extinction=nullptr,*phase=nullptr;
};
struct MieBracket {std::size_t lo=0;double weight=0;};
struct MieOpticalQuery {double diameter=0,temperature=0,mu=0;};
struct MieOpticalResult {double absorption=0,scattering=0,extinction=0,band=0,phase=0,asymmetry=0;};
UGKP_MIE_HD inline MieBracket mieLinearBracket(const double* axis,int size,double value){
    if(value<=axis[0])return {0,0};
    const std::size_t last=size-1;
    if(value>=axis[last])return {last-1,1};
    std::size_t first=0,count=size;
    while(count){const auto step=count/2,middle=first+step;
        if(!(value<axis[middle])){first=middle+1;count-=step+1;}else count=step;}
    const auto clipped=first<last?first:last;
    const auto hi=clipped<1?1:clipped;
    return {hi-1,(value-axis[hi-1])/(axis[hi]-axis[hi-1])};
}
UGKP_MIE_HD inline MieBracket mieLogBracket(const double* axis,int size,double value){
    if(!(value>0)||!std::isfinite(value))value=axis[0];
    auto result=mieLinearBracket(axis,size,value);
    // std::min/max argument order from the original table is significant for NaN.
    const double lower=axis[result.lo],upper=axis[result.lo+1];
    const double boundedUpper=value<upper?value:upper;
    const double bounded=lower<boundedUpper?boundedUpper:lower;
    const double a=std::log(lower),z=std::log(upper),v=std::log(bounded);
    result.weight=(v-a)/(z-a);return result;
}
UGKP_MIE_HD inline double mieBlend(double a,double b,double weight){return a+weight*(b-a);}
UGKP_MIE_HD inline double mieBilinear(const double* values,int nt,const MieBracket& d,const MieBracket& t){
    const auto a=d.lo*nt+t.lo,c=(d.lo+1)*nt+t.lo;
    return mieBlend(mieBlend(values[a],values[a+1],t.weight),mieBlend(values[c],values[c+1],t.weight),d.weight);
}
UGKP_MIE_HD inline double mieBand(const MieOpticalTableView& table,double temperature){
    const auto t=mieLinearBracket(table.temperature,table.temperatures,temperature);
    return mieBlend(table.band[t.lo],table.band[t.lo+1],t.weight);
}
UGKP_MIE_HD inline double miePhaseAt(const MieOpticalTableView& table,const MieBracket& d,
    const MieBracket& t,const MieBracket& m){
    const auto a=d.lo*table.temperatures+t.lo,b=a+1,c=(d.lo+1)*table.temperatures+t.lo,e=c+1;
    const double q=mieBilinear(table.scattering,table.temperatures,d,t);
    if(q==0)return 1;
    const double av=table.scattering[a]*mieBlend(table.phase[a*table.angles+m.lo],table.phase[a*table.angles+m.lo+1],m.weight);
    const double bv=table.scattering[b]*mieBlend(table.phase[b*table.angles+m.lo],table.phase[b*table.angles+m.lo+1],m.weight);
    const double cv=table.scattering[c]*mieBlend(table.phase[c*table.angles+m.lo],table.phase[c*table.angles+m.lo+1],m.weight);
    const double ev=table.scattering[e]*mieBlend(table.phase[e*table.angles+m.lo],table.phase[e*table.angles+m.lo+1],m.weight);
    return mieBlend(mieBlend(av,bv,t.weight),mieBlend(cv,ev,t.weight),d.weight)/q;
}
UGKP_MIE_HD inline double miePhase(const MieOpticalTableView& table,double diameter,double temperature,double mu){
    return miePhaseAt(table,mieLogBracket(table.diameter,table.diameters,diameter),
        mieLinearBracket(table.temperature,table.temperatures,temperature),mieLinearBracket(table.mu,table.angles,mu));
}
UGKP_MIE_HD inline MieOpticalResult mieOpticalProperties(const MieOpticalTableView& table,const MieOpticalQuery& query){
    const auto d=mieLogBracket(table.diameter,table.diameters,query.diameter);
    const auto t=mieLinearBracket(table.temperature,table.temperatures,query.temperature);
    MieOpticalResult result;
    result.absorption=mieBilinear(table.absorption,table.temperatures,d,t);
    result.scattering=mieBilinear(table.scattering,table.temperatures,d,t);
    result.extinction=mieBilinear(table.extinction,table.temperatures,d,t);
    result.band=mieBand(table,query.temperature);
    result.phase=miePhaseAt(table,d,t,mieLinearBracket(table.mu,table.angles,query.mu));
    double denominator=0,moment=0;
    double previous=miePhaseAt(table,d,t,mieLinearBracket(table.mu,table.angles,table.mu[0]));
    for(int i=1;i<table.angles;++i){
        const double current=miePhaseAt(table,d,t,mieLinearBracket(table.mu,table.angles,table.mu[i]));
        const double delta=table.mu[i]-table.mu[i-1];
        denominator+=.5*(previous+current)*delta;
        moment+=.5*(table.mu[i-1]*previous+table.mu[i]*current)*delta;previous=current;
    }
    const double value=moment/denominator;
    const double upper=1<value?1:value;
    result.asymmetry=-1>upper?-1:upper;
    return result;
}
}}
#undef UGKP_MIE_HD
#endif
