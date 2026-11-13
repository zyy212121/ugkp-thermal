#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKWP_PARTICLE_CALORIC_HD __host__ __device__
#else
#define UGKWP_PARTICLE_CALORIC_HD
#endif
namespace ugkwp {
// Single binary32 caloric state and arithmetic.
// The caller selects the existing ordinary or active cold-profile law.
struct ParticleCaloric {
 bool phaseChange=false;
 float meltingTemperature=2327.0f, mushyRange=20.0f, latentHeat=0.0f, solidCp=0.0f;
 UGKWP_PARTICLE_CALORIC_HD static float liquidCp(float t) {
  t=t<2327.0f?2327.0f:(t>4000.0f?4000.0f:t);
  return (153.5f+0.0031f*(t-2327.0f))/0.101961276f;
 }
 UGKWP_PARTICLE_CALORIC_HD static float liquidH(float t) {
  const float cp=liquidCp(2327.0f), slope=0.0031f/0.101961276f;
  if(t<=2327.0f)return cp*t;
  const float d=t<4000.0f?t-2327.0f:1673.0f;
  return cp*2327.0f+cp*d+0.5f*slope*d*d+(t>4000.0f?liquidCp(4000.0f)*(t-4000.0f):0.0f);
 }
 UGKWP_PARTICLE_CALORIC_HD float solidus()const{return meltingTemperature-0.5f*mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD float liquidus()const{return meltingTemperature+0.5f*mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD float mushyCp()const{return 0.5f*(solidCp+liquidCp(meltingTemperature))+latentHeat/mushyRange;}
 UGKWP_PARTICLE_CALORIC_HD float enthalpy(float t)const {
  if(!phaseChange)return liquidH(t);
  const float ts=solidus(),tl=liquidus(),hs=solidCp*ts;
  if(t<=ts)return solidCp*t;
  if(t<=tl)return hs+mushyCp()*(t-ts);
  // Same existing cold-wall caloric law, including its endpoint trapezoid.
  return hs+mushyCp()*mushyRange+0.5f*(liquidCp(tl)+liquidCp(t))*(t-tl);
 }
 UGKWP_PARTICLE_CALORIC_HD float specificHeat(float t)const {
  if(!phaseChange)return liquidCp(t);
  return t<=solidus()?solidCp:(t<liquidus()?mushyCp():liquidCp(t));
 }
 // Derivative of the stored enthalpy law. Above the phase liquidus the
 // existing endpoint trapezoid is authoritative, including outside the
 // tabulated liquid-Cp interval; specificHeat() retains its original meaning.
 UGKWP_PARTICLE_CALORIC_HD float enthalpyDerivative(float t)const {
  if(!phaseChange)return liquidCp(t);
  if(t<=solidus())return solidCp;
  if(t<liquidus())return mushyCp();
  const float slope=t>=2327.f&&t<4000.f?0.0031f/0.101961276f:0.f;
  return .5f*(liquidCp(liquidus())+liquidCp(t))+.5f*slope*(t-liquidus());
 }
 // Exact integral of the existing piecewise-linear heat-capacity law,
 // evaluated on the small interval rather than subtracting large H values.
 UGKWP_PARTICLE_CALORIC_HD float enthalpyIncrement(float t,float deltaT)const {
  if(deltaT==0.f)return 0.f;
  const bool reverse=deltaT<0.f;
  float current=t,remaining=reverse?-deltaT:deltaT,total=0.f;
  const float slope=0.0031f/0.101961276f;
  for(int segment=0;segment<8 && remaining>0.f;++segment){
   float width=remaining;
   const float boundaries[4]={2327.f,4000.f,phaseChange?solidus():2327.f,phaseChange?liquidus():4000.f};
   for(int k=0;k<4;++k){
    const float distance=reverse?current-boundaries[k]:boundaries[k]-current;
    if(distance>0.f&&distance<width)width=distance;
   }
   const float side=reverse?::nextafterf(current,-3.4028234663852886e38f):current;
   float derivative=liquidCp(current),curvature=side>=2327.f&&side<4000.f?slope:0.f;
   if(phaseChange){
    if(side<solidus()){derivative=solidCp;curvature=0.f;}
    else if(side<liquidus()){derivative=mushyCp();curvature=0.f;}
    else derivative=.5f*(liquidCp(liquidus())+liquidCp(current))+.5f*curvature*(current-liquidus());
   }
   total+=width*(derivative+(reverse?-.5f:.5f)*curvature*width);
   current+=reverse?-width:width;remaining-=width;
  }
  return reverse?-total:total;
 }
 UGKWP_PARTICLE_CALORIC_HD float temperature(float h)const {
  const float cp=liquidCp(2327.0f),slope=0.0031f/0.101961276f;
  if(!phaseChange){
   const float hlo=cp*2327.0f,hhi=liquidH(4000.0f);
   if(h<=hlo)return h/cp;
   if(h>=hhi)return 4000.0f+(h-hhi)/liquidCp(4000.0f);
   const float dh=h-hlo;
   return 2327.0f+2.0f*dh/(cp+std::sqrt(cp*cp+2.0f*slope*dh));
  }
  const float ts=solidus(),tl=liquidus(),hs=solidCp*ts,hl=hs+mushyCp()*mushyRange;
  if(h<=hs)return h/solidCp;
  if(h<=hl)return ts+(h-hs)/mushyCp();
  // Invert each linear/quadratic segment of the existing trapezoidal law.
  float lo=tl;
  for(int k=0;k<3;++k){
   const float hi=lo<2327.0f?2327.0f:(lo<4000.0f?4000.0f:3.4028234663852886e38f);
   if(hi<3.4028234663852886e38f && h>enthalpy(hi)){lo=hi;continue;}
   const float d=h-enthalpy(lo);
   const float s=lo>=2327.0f&&lo<4000.0f?slope:0.0f;
   const float b=0.5f*(liquidCp(tl)+liquidCp(lo))+0.5f*s*(lo-tl);
   return lo+(s==0.0f?d/b:2.0f*d/(b+std::sqrt(b*b+2.0f*s*d)));
  }
  return lo;
 }
};
}
#undef UGKWP_PARTICLE_CALORIC_HD
