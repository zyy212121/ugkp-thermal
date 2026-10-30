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
// Available inventory / requested outgoing inventory, without forming either
// density*volume or dt*outflow. Those products may underflow even though the
// subsequent density update (dt/volume)*outflow is representable.
// The caller supplies finite nonnegative inventories and positive geometry.
// Keep the existing 0.999 reserve when depletion would otherwise occur.
template<class R>
UGKP_INVENTORY_HD inline R inventoryOutflowScale(R density,R volume,R dt,R outflow)
{
    if(!(dt>R(0)) || !(outflow>R(0)))return R(1);
    if(density<=R(0))return R(0);
    int ed,ev,et,ef;
    const R md=R(::frexp(density,&ed)),mv=R(::frexp(volume,&ev));
    const R mt=R(::frexp(dt,&et)),mf=R(::frexp(outflow,&ef));
    const R fraction=R(::ldexp((md*mv)/(mt*mf),ed+ev-et-ef));
    if(!(fraction<R(1)))return R(1);
    const R scale=R(.999)*fraction;
    // At subnormal scale the reserve itself can round away. Choose the next
    // representable value towards zero to retain a conservative scale.
    if constexpr(std::is_same<R,float>::value)return ::nextafterf(scale,0.f);
    else return R(::nextafter(double(scale),0.));
}

// The ratio alone is insufficient when the scaled face flux is subnormal:
// nearest rounding can increase a requested 0.75 quantum to one full quantum.
// After multiplication by dt/V that can exceed the donor's entire inventory.
// Bias only these limited, underflow-range products towards zero. The shared
// face value still enters its two cell balances with opposite signs; normal
// flux arithmetic and an inactive limiter are unchanged.
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
