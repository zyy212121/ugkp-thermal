#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKWP_PAIRED_HEAT_HD __host__ __device__
#else
#define UGKWP_PAIRED_HEAT_HD
#endif
namespace ugkwp {
UGKWP_PAIRED_HEAT_HD inline float pairedHeatBathFraction(float a,float C,float Cg)
{
 if(!(a>0.f)||!(C>0.f)||!(Cg>0.f))return 1.f;
 if(a>1.f)a=1.f;
 const float invR=Cg>=C?1.f/(1.f+C/Cg):(Cg/C)/(1.f+Cg/C);
 if(a==1.f||invR==0.f)return invR;
 const float d=-::log1p(-a),x=d/invR;
 // Evaluate tiny time/capacity ratios without cancellation or overflowing1/a.
 float beta=x<1e-5f?(1.f-x*.5f+x*x/6.f)*(d/a)
  :(invR/a)*(-::expm1(-x));
 return beta<0.f?0.f:(beta>1.f?1.f:beta);
}
UGKWP_PAIRED_HEAT_HD inline float pairedHeatBath(float A,float B,float C,float Cg,float Tg)
{
 if(!(A>0.f)||!(C>0.f))return Tg;
 const float mean=B/A,beta=pairedHeatBathFraction(A/C,C,Cg);
 return mean+beta*(Tg-mean);
}
UGKWP_PAIRED_HEAT_HD inline float pairedHeatAcceptanceScale(float proposed,float available)
{
 if(!(proposed>0.f))return 1.f;
 if(!(available>0.f))return 0.f;
 return proposed>available?available/proposed:1.f;
}
template<class Caloric>
UGKWP_PAIRED_HEAT_HD inline float boundedParticleSpecificHeat
(const Caloric& law,float oldH,float Ti,float Tg,float /*cp*/,float a,float Tmin,float Tmax)
{
 if(!(a>0.f)||Ti==Tg)return 0.f;
 if(a>1.f)a=1.f;
 float next=Ti+a*(Tg-Ti);
 next=next<Tmin?Tmin:(next>Tmax?Tmax:next);
 return law.enthalpy(next)-oldH;
}
}
#undef UGKWP_PAIRED_HEAT_HD
