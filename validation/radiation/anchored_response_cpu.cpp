#include "common/gpu/AnchoredParticleRadiationResponse.H"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#ifdef TEST_FP32
using Real=float;
#else
using Real=double;
#endif
struct Law {Real enthalpy(Real t)const{return Real(2)*t;}Real temperature(Real h)const{return h/Real(2);}};
using Entry=ugkp::gpu::RadiationEntryParticle<Real>;
using Response=ugkp::gpu::AnchoredParticleRadiationResponse<Real,Law>;
bool near(Real a,Real b){return std::abs(a-b)<=Real(32)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b));}
int main(){
 Entry p[3]{};p[0].mass=Real(.25);p[0].thermal.temperature=Real(300);p[0].eligible=true;
 p[1].mass=Real(.75);p[1].thermal.temperature=Real(700);p[1].eligible=true;
 p[2].mass=Real(100);p[2].thermal.temperature=Real(999);p[2].eligible=false;
#ifndef TEST_FP32
 p[0].thermal.specificEnthalpyResidual=.03125;p[1].thermal.specificEnthalpyResidual=-.0625;
#endif
 Entry snapshot[3];std::memcpy(snapshot,p,sizeof(p));Response response{p,3,Real(100),Real(1000),{}};
 auto d=response.domain();assert(d.lower<Real(0)&&d.upper>Real(0)&&d.mass==Real(1));
 for(Real q:{Real(0),d.upper/Real(2),d.lower/Real(3),d.upper,d.lower}){
  auto r=response(q);auto cached=response.evaluate(q,d);assert(cached.requestedEnergy==r.requestedEnergy&&cached.actualEnergy==r.actualEnergy&&cached.temperature==r.temperature&&cached.fraction==r.fraction&&cached.withinDomain==r.withinDomain&&cached.publicationValid==r.publicationValid);assert(r.withinDomain&&r.publicationValid&&near(r.actualEnergy,q));
  Real actual=0,mean=0;
  for(int i=0;i<2;++i){auto c=p[i].thermal;auto pub=ugkwp::publishParticleThermalHeat(c,r.fraction*response.specificCapacity(p[i],q>=0),Real(100),Real(1000),Law{});
   actual+=p[i].mass*pub.acceptedSpecificHeat;mean+=p[i].mass*ugkwp::effectiveParticleTemperature(c,Law{});}
  assert(actual==r.actualEnergy&&mean==r.temperature);
 }
 assert(!response(d.upper+Real(1)).withinDomain);assert(!response(d.lower-Real(1)).withinDomain);
 // Sign changes and repetitions are from the same entrance, never cumulative.
 auto a=response(d.upper/Real(4));(void)response(d.lower/Real(2));auto b=response(d.upper/Real(4));
 assert(a.actualEnergy==b.actualEnergy&&a.temperature==b.temperature);assert(std::memcmp(p,snapshot,sizeof(p))==0);
 Response empty{nullptr,0,Real(100),Real(1000),{}};assert(empty(Real(0)).withinDomain&&!empty(Real(1)).withinDomain);
#ifndef TEST_FP32
 // A sub-temperature-ULP heat remains represented by the existing FP64 carry.
 auto tiny=response(Real(1e-14));assert(tiny.publicationValid&&near(tiny.actualEnergy,Real(1e-14)));
#endif
 std::cout<<"ANCHOR_PASS bits="<<sizeof(Real)*8<<" heating cooling endpoints out_of_domain repeated_sign_change immutable_entry inactive empty actual_publication\n";
}
