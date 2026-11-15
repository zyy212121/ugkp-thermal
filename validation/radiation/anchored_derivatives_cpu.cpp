#include "common/gpu/AnchoredRadiationDerivatives.H"
#include "common/GpuParticleCaloric.H"
#include <cassert>
#include <iostream>
#ifdef TEST_FP32
using Real=float;
#else
using Real=double;
#endif
bool near(Real a,Real b,Real tol=Real(2e-5)){return std::abs(a-b)<=tol*std::max(Real(1e-20),std::abs(b));}
int main(){using namespace ugkp::gpu;
 RadiationEntryParticle<Real> p[2]{};for(int i=0;i<2;++i){p[i].mass=Real(i+1);p[i].thermal.temperature=Real(i?3500:2500);p[i].eligible=true;}
 ugkwp::ParticleCaloric law;AnchoredParticleRadiationResponse<Real,ugkwp::ParticleCaloric> response{p,2,Real(100),Real(5000),law};
 for(bool heating:{false,true}){auto d=anchoredTemperatureDerivative(response,heating);assert(d.valid&&d.available&&d.derivative>Real(0));
  auto dom=response.domain();Real sum=0;for(int i=0;i<2;++i)sum+=p[i].mass*std::abs(response.specificCapacity(p[i],heating))/law.enthalpyDerivative(p[i].thermal.temperature);
  assert(near(d.derivative,sum/(dom.mass*(heating?dom.upper:-dom.lower))));}
 // At solidus/liquidus use the physical one-sided H' rather than specificHeat.
 law.phaseChange=true;law.solidCp=Real(800);law.latentHeat=Real(300000);response.caloric=law;response.count=1;
 for(Real boundary:{Real(law.solidus()),Real(law.liquidus()),Real(4000)}){
  p[0].thermal.temperature=boundary;
  for(bool heating:{false,true}){auto d=anchoredTemperatureDerivative(response,heating);
   Real side=std::nextafter(boundary,heating?std::numeric_limits<Real>::infinity():-std::numeric_limits<Real>::infinity());
   assert(d.valid&&near(d.derivative,Real(1)/(p[0].mass*law.enthalpyDerivative(side))));}}
 p[0].thermal.temperature=Real(100);assert(!anchoredTemperatureDerivative(response,false).available);
 const Real axis[]={Real(100),Real(200),Real(300)},band[]={Real(.2),Real(.4),Real(.3)};
 for(Real t:{Real(50),Real(100),Real(150),Real(200),Real(250),Real(300),Real(400)})for(bool heating:{false,true}){
  auto d=bandEmissionDerivative(axis,band,3,t,heating);assert(d.valid);
  Real slope= t<Real(100)||t>Real(300)||(t==Real(100)&&!heating)||(t==Real(300)&&heating)?Real(0):((t<Real(200)||(t==Real(200)&&!heating))?Real(.002):Real(-.001));
  assert(near(d.bandDerivative,slope));
  Real expected=Real(4)*Real(5.670374419e-8)*(slope*t*t*t*t+Real(4)*d.band*t*t*t);assert(near(d.derivative,expected));
 }
 const Real steep[]={Real(1),Real(0),Real(0)};auto negative=bandEmissionDerivative(axis,steep,3,Real(190),true);assert(negative.valid&&negative.derivative<0);
 assert(!bandEmissionDerivative<Real>(nullptr,nullptr,0,Real(100),true).valid);
 std::cout<<"DERIVATIVE_PASS bits="<<sizeof(Real)*8<<" bidirectional production_Hprime phase_junctions exhausted_capacity mie_knots extrapolation T4 signed_slope\n";
}
