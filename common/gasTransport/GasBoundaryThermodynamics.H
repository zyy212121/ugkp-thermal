#pragma once
#if defined(__CUDACC__)
#define UGKP_BOUNDARY_THERMO_HD __host__ __device__
#else
#define UGKP_BOUNDARY_THERMO_HD
#endif
namespace ugkwp {
// Constraint precedence shared by mixture face flux and wave-pressure speed.
// The caller validates the resulting EOS state with its existing diagnostics.
template<class Real> struct BoundaryThermodynamics {Real rho,pressure,temperature;};
template<class Real>
UGKP_BOUNDARY_THERMO_HD inline BoundaryThermodynamics<Real> mixtureBoundaryThermodynamics(
    BoundaryThermodynamics<Real> owner,BoundaryThermodynamics<Real> boundary,
    bool fixedRho,bool fixedPressure,bool fixedTemperature,Real gasConstant)
{
    if(fixedRho)owner.rho=boundary.rho;
    if(fixedPressure)owner.pressure=boundary.pressure;
    if(fixedTemperature)owner.temperature=boundary.temperature;
    if(fixedRho&&fixedPressure&&!fixedTemperature)
        owner.temperature=owner.pressure/(owner.rho*gasConstant);
    else if(fixedRho&&!fixedPressure)
        owner.pressure=owner.rho*gasConstant*owner.temperature;
    return owner;
}
}
#undef UGKP_BOUNDARY_THERMO_HD
