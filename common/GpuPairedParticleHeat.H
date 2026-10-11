#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKWP_PAIRED_HEAT_HD __host__ __device__
#else
#define UGKWP_PAIRED_HEAT_HD
#endif
namespace ugkwp {
UGKWP_PAIRED_HEAT_HD inline double pairedHeatBathFraction(double a,double C,double Cg)
{
 if(!(a>0.)||!(C>0.)||!(Cg>0.))return 1.;
 if(a>1.)a=1.;
 const double invR=Cg>=C?1./(1.+C/Cg):(Cg/C)/(1.+Cg/C);
 if(a==1.||invR==0.)return invR;
 const double d=-::log1p(-a),x=d/invR;
 // Evaluate tiny time/capacity ratios without cancellation or overflowing1/a.
 double beta=x<1e-5?(1.-x*.5+x*x/6.)*(d/a)
  :(invR/a)*(-::expm1(-x));
 return beta<0.?0.:(beta>1.?1.:beta);
}
UGKWP_PAIRED_HEAT_HD inline double pairedHeatBath(double A,double B,double C,double Cg,double Tg)
{
 if(!(A>0.)||!(C>0.))return Tg;
 const double mean=B/A,beta=pairedHeatBathFraction(A/C,C,Cg);
 return mean+beta*(Tg-mean);
}
UGKWP_PAIRED_HEAT_HD inline double pairedHeatAcceptanceScale(double proposed,double available)
{
 if(!(proposed>0.))return 1.;
 if(!(available>0.))return 0.;
 return proposed>available?available/proposed:1.;
}
template<class Caloric>
UGKWP_PAIRED_HEAT_HD inline double boundedParticleSpecificHeat
(const Caloric& law,double oldH,double Ti,double Tg,double /*cp*/,double a,double Tmin,double Tmax)
{
 if(!(a>0.)||Ti==Tg)return 0.;
 if(a>1.)a=1.;
 double next=Ti+a*(Tg-Ti);
 next=next<Tmin?Tmin:(next>Tmax?Tmax:next);
 return law.enthalpy(next)-oldH;
}
}
#undef UGKWP_PAIRED_HEAT_HD
