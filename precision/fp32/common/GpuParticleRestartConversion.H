#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKWP_PARTICLE_CALORIC_HD __host__ __device__
#else
#define UGKWP_PARTICLE_CALORIC_HD
#endif
namespace ugkwp { namespace legacyRestart {


struct ParticleCaloric {
 bool phaseChange=false;
 double meltingTemperature=2327.0, mushyRange=20.0, latentHeat=0.0, solidCp=0.0;
 UGKWP_PARTICLE_CALORIC_HD static double liquidCp(double t) {
  t=t<2327.0?2327.0:(t>4000.0?4000.0:t);
  return (153.5+0.0031*(t-2327.0))/0.101961276;
 }
 UGKWP_PARTICLE_CALORIC_HD static double liquidH(double t) {
  const double cp=liquidCp(2327.0), slope=0.0031/0.101961276;
  if(t<=2327.0)return cp*t;
  const double d=t<4000.0?t-2327.0:1673.0;
  return cp*2327.0+cp*d+0.5*slope*d*d+(t>4000.0?liquidCp(4000.0)*(t-4000.0):0.0);
 }
 UGKWP_PARTICLE_CALORIC_HD double solidus()const{return meltingTemperature-0.5*mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD double liquidus()const{return meltingTemperature+0.5*mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD double mushyCp()const{return 0.5*(solidCp+liquidCp(meltingTemperature))+latentHeat/mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD double enthalpy(double t)const {
  if(!phaseChange)return liquidH(t);
  const double ts=solidus(),tl=liquidus(),hs=solidCp*ts;
  if(t<=ts)return solidCp*t;
  if(t<=tl)return hs+mushyCp()*(t-ts);

  return hs+mushyCp()*mushyRange+0.5*(liquidCp(tl)+liquidCp(t))*(t-tl);
 }
 UGKWP_PARTICLE_CALORIC_HD double specificHeat(double t)const {
  if(!phaseChange)return liquidCp(t);
  return t<=solidus()?solidCp:(t<liquidus()?mushyCp():liquidCp(t));
 }



 UGKWP_PARTICLE_CALORIC_HD double enthalpyDerivative(double t)const {
  if(!phaseChange)return liquidCp(t);
  if(t<=solidus())return solidCp;
  if(t<liquidus())return mushyCp();
  const double slope=t>=2327.&&t<4000.?0.0031/0.101961276:0.;
  return .5*(liquidCp(liquidus())+liquidCp(t))+.5*slope*(t-liquidus());
 }


 UGKWP_PARTICLE_CALORIC_HD double enthalpyIncrement(double t,double deltaT)const {
  if(deltaT==0.)return 0.;
  const bool reverse=deltaT<0.;
  double current=t,remaining=reverse?-deltaT:deltaT,total=0.;
  const double slope=0.0031/0.101961276;
  for(int segment=0;segment<8 && remaining>0.;++segment){
   double width=remaining;
   const double boundaries[4]={2327.,4000.,phaseChange?solidus():2327.,phaseChange?liquidus():4000.};
   for(int k=0;k<4;++k){
    const double distance=reverse?current-boundaries[k]:boundaries[k]-current;
    if(distance>0.&&distance<width)width=distance;
   }
   const double side=reverse?::nextafter(current,-1.7976931348623157e308):current;
   double derivative=liquidCp(current),curvature=side>=2327.&&side<4000.?slope:0.;
   if(phaseChange){
    if(side<solidus()){derivative=solidCp;curvature=0.;}
    else if(side<liquidus()){derivative=mushyCp();curvature=0.;}
    else derivative=.5*(liquidCp(liquidus())+liquidCp(current))+.5*curvature*(current-liquidus());
   }
   total+=width*(derivative+(reverse?-.5:.5)*curvature*width);
   current+=reverse?-width:width;remaining-=width;
  }
  return reverse?-total:total;
 }
 UGKWP_PARTICLE_CALORIC_HD double temperature(double h)const {
  const double cp=liquidCp(2327.0),slope=0.0031/0.101961276;
  if(!phaseChange){
   const double hlo=cp*2327.0,hhi=liquidH(4000.0);
   if(h<=hlo)return h/cp;
   if(h>=hhi)return 4000.0+(h-hhi)/liquidCp(4000.0);
   const double dh=h-hlo;
   return 2327.0+2.0*dh/(cp+::sqrt(cp*cp+2.0*slope*dh));
  }
  const double ts=solidus(),tl=liquidus(),hs=solidCp*ts,hl=hs+mushyCp()*mushyRange;
  if(h<=hs)return h/solidCp;
  if(h<=hl)return ts+(h-hs)/mushyCp();

  double lo=tl;
  for(int k=0;k<3;++k){
   const double hi=lo<2327.0?2327.0:(lo<4000.0?4000.0:1.7976931348623157e308);
   if(hi<1e300 && h>enthalpy(hi)){lo=hi;continue;}
   const double d=h-enthalpy(lo);
   const double s=lo>=2327.0&&lo<4000.0?slope:0.0;
   const double b=0.5*(liquidCp(tl)+liquidCp(lo))+0.5*s*(lo-tl);
   return lo+(s==0.0?d/b:2.0*d/(b+::sqrt(b*b+2.0*s*d)));
  }
  return lo;
 }
};
}}
#undef UGKWP_PARTICLE_CALORIC_HD

#include "GpuParticleCaloric.H"
namespace ugkwp { namespace legacyRestart {


inline float importMobileTemperature(double temperature,double carry) {
 if(carry==0.0)return static_cast<float>(temperature);
 const ParticleCaloric law;
 return static_cast<float>(law.temperature(law.enthalpy(temperature)+carry));
}
inline void importProfile(const float* oldNodes,float* nodes,int count,double carry) {
 for(int n=0;n<count;++n)nodes[n]=static_cast<float>(double(oldNodes[n])+carry);
}
template<class Snapshot>
inline void importParticleThermalHistory(Snapshot& p,double originalTemperature,
 double carry,int contactModel,const ParticleCaloric& phaseLaw) {
 const int count=p.state?(contactModel==3?8:(contactModel==4?64:0)):0;
 if(!count){p.thermal.temperature=importMobileTemperature(originalTemperature,carry);return;}
 float* nodes=count==8?p.thermal.coldNodeSpecificEnthalpy:p.thermal.cold2DNodeSpecificEnthalpy;
 importProfile(nodes,nodes,count,carry);
 float mean=0.0f;for(int n=0;n<count;++n)mean+=nodes[n]/float(count);
 ugkwp::ParticleCaloric law;
 law.phaseChange=true;law.meltingTemperature=float(phaseLaw.meltingTemperature);
 law.mushyRange=float(phaseLaw.mushyRange);law.latentHeat=float(phaseLaw.latentHeat);
 law.solidCp=float(phaseLaw.solidCp);
 p.thermal.temperature=law.temperature(mean);
}
}}
