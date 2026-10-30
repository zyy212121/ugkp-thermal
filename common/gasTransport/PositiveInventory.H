#pragma once
#include <cmath>
#include <type_traits>
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
    // representable value towards zero so the face never overdraws its donor.
    if constexpr(std::is_same<R,float>::value)return ::nextafterf(scale,0.f);
    else return R(::nextafter(double(scale),0.));
}
}
#undef UGKP_INVENTORY_HD
