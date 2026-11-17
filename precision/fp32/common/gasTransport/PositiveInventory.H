#pragma once
#include <cmath>
#include <type_traits>
#include <limits>
#if defined(__CUDACC__)
#define UGKP_INVENTORY_HD __host__ __device__
#else
#define UGKP_INVENTORY_HD
#endif
namespace ugkwp {





template<class R,class Volume,class Time>
UGKP_INVENTORY_HD inline R inventoryOutflowScale(R density,Volume volume,Time dt,R outflow)
{
    if(!(dt>R(0)) || !(outflow>R(0)))return R(1);
    if(density<=R(0))return R(0);
    int ed,ev,et,ef;
    const R md=R(::frexp(density,&ed)),mv=R(::frexp(volume,&ev));
    const R mt=R(::frexp(dt,&et)),mf=R(::frexp(outflow,&ef));
    const R fraction=R(::ldexp((md*mv)/(mt*mf),ed+ev-et-ef));
    if(!(fraction<R(1)))return R(1);
    const R scale=R(.999)*fraction;


    if constexpr(std::is_same<R,float>::value)return ::nextafterf(scale,0.f);
    else return R(::nextafter(double(scale),0.));
}







template<class R>
UGKP_INVENTORY_HD inline R scaleInventoryFlux(R flux,R factor)
{
    const R scaled=flux*factor;
    if(factor>R(0) && factor<R(1) && scaled!=R(0)
        && ::fabs(scaled)<=std::numeric_limits<R>::min())
    {
        if constexpr(std::is_same<R,float>::value)return ::nextafterf(scaled,0.f);
        else return R(::nextafter(double(scaled),0.));
    }
    return scaled;
}

}
#undef UGKP_INVENTORY_HD
