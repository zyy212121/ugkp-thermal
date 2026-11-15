#pragma once
#include <cmath>
#include <limits>
#if defined(__CUDACC__)
#define UGKP_RES_HD __host__ __device__
#else
#define UGKP_RES_HD
#endif
namespace ugkp { namespace gpu {
// A per-endpoint one-temperature-storage-grid allowance. Exchange assessment
// receives the sum of entrance and output allowances. This is not a
// claim that the full thermal state (which may carry enthalpy) is quantized
// by temperature alone, nor a proof of the best shared-fraction candidate.
template<class Real,class Caloric> UGKP_RES_HD Real publicationTemperatureGrid(
 Real t,Real lo,Real hi,const Caloric& law) {
 const Real h=law.enthalpy(t);
 const Real up=std::nextafter(t,Real(INFINITY));
 const Real dn=std::nextafter(t,-Real(INFINITY));
 const Real a=up<=hi?std::abs(Real(law.enthalpy(up))-h):Real(0);
 const Real b=dn>=lo?std::abs(h-Real(law.enthalpy(dn))):Real(0);
 return a>b?a:b;
}
template<class Real> struct MaterialConvergenceAssessment {
 Real residual=0,rawNormalized=0,temperatureGridAllowance=0,allowedResidual=0;
 Real effectiveNormalized=0;
 bool valid=false,accepted=false,usedTemperatureGrid=false;
};
template<class Real> UGKP_RES_HD MaterialConvergenceAssessment<Real>
 assessMaterialPublication(Real actual,Real physical,Real temperatureGrid,Real relativeTolerance){
 MaterialConvergenceAssessment<Real> a;
 a.residual=actual-physical;a.temperatureGridAllowance=temperatureGrid;
 a.valid=std::isfinite(actual)&&std::isfinite(physical)&&std::isfinite(a.residual)
  &&std::isfinite(temperatureGrid)&&temperatureGrid>=Real(0)
  &&std::isfinite(relativeTolerance)&&relativeTolerance>Real(0);
 if(!a.valid){a.rawNormalized=a.effectiveNormalized=Real(INFINITY);return a;}
 const Real absolute=std::abs(a.residual);
 const Real scale=std::abs(actual)>std::abs(physical)?std::abs(actual):std::abs(physical);
 a.rawNormalized=scale>Real(0)?absolute/scale:Real(0);
 const Real requested=relativeTolerance*scale;
 a.allowedResidual=temperatureGrid>requested?temperatureGrid:requested;
 a.accepted=absolute<=a.allowedResidual;
 a.usedTemperatureGrid=a.accepted&&absolute>requested;
 a.effectiveNormalized=a.rawNormalized;
 if(temperatureGrid>Real(0)){
  const Real scaled=(absolute/temperatureGrid)*relativeTolerance;
  if(scaled<a.effectiveNormalized)a.effectiveNormalized=scaled;
 }
 return a;
}
}}
#undef UGKP_RES_HD
