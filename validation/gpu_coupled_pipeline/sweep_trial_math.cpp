#include "../../common/gpu/SweepTrialMath.H"
#include <random>
#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>
#ifdef TEST_DEVICE_SWEEP_TRIAL
#include "sweep_trial_probe.H"
#endif
namespace g=ugkp::gpu::geometry;
int main(){std::size_t cases=0;std::mt19937_64 random(961341);
#ifdef TEST_DEVICE_SWEEP_TRIAL
    std::vector<SweepTrialInput> inputs;std::vector<g::SweepTrialCoordinate> results;
#endif
    const auto check=[&](double origin,double correction,double alpha){
        const long double exact=static_cast<long double>(origin)+static_cast<long double>(alpha)*correction;
        const bool expected=std::isfinite(exact)&&std::abs(exact)<=std::numeric_limits<double>::max();
        const bool got=g::sweepTrialRangeFp64(origin,correction,alpha);
        if(got!=expected){std::cerr.precision(17);std::cerr<<"range differs "<<origin<<' '<<correction<<' '<<alpha<<'\n';std::abort();}
        const auto result=g::sweepTrialCoordinate(origin,correction,alpha);assert(result.representable==expected);
#ifdef TEST_DEVICE_SWEEP_TRIAL
        inputs.push_back({origin,correction,alpha});results.push_back(result);
#endif
        if(expected){const double next=origin+alpha*correction;assert(std::memcmp(&next,&result.value,sizeof(next))==0);assert(result.changed==(next!=origin));}++cases;
    };
    const double maximum=std::numeric_limits<double>::max();
    std::vector<double> origins={0.,-0.,maximum,-maximum,std::nextafter(maximum,0.),-std::nextafter(maximum,0.)};
    for(int power=850;power<1024;++power)for(int backtrack:{0,1,10,50,500,1074})for(double origin:origins)for(double sign:{-1.,1.}){
        const double correction=sign*std::ldexp(1.,power);for(double value:{std::nextafter(correction,0.),correction,std::nextafter(correction,sign*std::numeric_limits<double>::infinity())})check(origin,value,std::ldexp(1.,-backtrack));}
    for(int i=0;i<100000;++i){auto a=random(),b=random();double origin,correction;std::memcpy(&origin,&a,sizeof(a));std::memcpy(&correction,&b,sizeof(b));check(origin,correction,std::ldexp(1.,-int(random()%1075)));}
#ifdef TEST_DEVICE_SWEEP_TRIAL
    const auto actual=probeSweepTrial(inputs);assert(actual.size()==results.size());
    for(std::size_t i=0;i<actual.size();++i){assert(actual[i].representable==results[i].representable);assert(actual[i].changed==results[i].changed);
        if(results[i].representable)assert(std::memcmp(&actual[i].value,&results[i].value,sizeof(double))==0);}
#endif
    std::cout<<"PASS "<<cases<<" native long-double/FP64 range decisions and represented coordinate updates\n";
}
