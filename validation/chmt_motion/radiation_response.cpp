#include "gpu/ParticleRadiationResponse.H"
#include "GpuParticleThermalCarry.H"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
int main(){
 using namespace ugkp::gpu;const ugkwp::ParticleCaloric law{};
 int count=3,cell[]={0,0,0},status[]={1,1,1};unsigned char wall[]={0,0,1};
 double mass[]={.3,.7,10},temperature[]={1000,2500,3500},carry[]={1e-5,-3e-5,0};
 ParticleRadiationResponseView view{true,3,&count,cell,status,nullptr,wall,mass,temperature,carry,1,5000};
 IndexedParticleRadiationResponse response{view,nullptr,0,3};
 int cases=0;
 for(bool uniform:{true,false})for(double eq:{400.,4000.,8000.})for(double dt:{.01,1.,10.,100.,1000.}){
  temperature[0]=1000;temperature[1]=uniform?1000:2500;
  carry[0]=uniform?0:1e-5;carry[1]=uniform?0:-3e-5;
  const auto power=[&](double t){return 1e-8*(std::pow(eq,4)-std::pow(t,4));};
  const auto root=solveCaloricParticleRadiation(response,power,dt);
  const bool heating=power(response(0,true).temperature)>0;
  const auto limit=response(1,heating);const double fraction=limit.energy!=0?root.energy/limit.energy:0;
  double actual=0,mean=0;
  for(int i=0;i<2;++i){ugkwp::ParticleThermalCarry<double> state{temperature[i],carry[i]};
   const double h=ugkwp::effectiveParticleEnthalpy(state,law);
   const double delta=radiationSpecificCapacity(h,law.enthalpy(1),law.enthalpy(5000),heating);
   auto accepted=ugkwp::publishParticleThermalHeat(state,fraction*delta,1.,5000.,law);assert(accepted.valid);
   actual+=mass[i]*accepted.acceptedSpecificHeat;mean+=mass[i]*ugkwp::effectiveParticleTemperature(state,law);
  }
  assert(std::abs(actual-root.energy)<1e-7);assert(std::abs(mean-root.temperature)<1e-8);
  const double residual=actual-dt*power(mean),scale=std::max(1.,std::max(std::abs(actual),std::abs(dt*power(mean))));
  if(root.fraction<1){assert(std::abs(residual)<1e-8*scale);assert((heating&&mean<=eq+1e-8)||(!heating&&mean>=eq-1e-8));}
  else assert(std::abs(mean-5000)<1e-8);
  if(uniform&&eq==4000&&dt==100){assert(std::abs(mean-3982.1811218380944)<1e-6);std::cout<<std::setprecision(17)<<"COUNTEREXAMPLE_FIXED T="<<mean<<" Q="<<actual<<" residual="<<residual<<"\n";}
  ++cases;
 }
 // Fixed-capacity compatibility is the linear limit of the same implicit root.
 struct Linear {CaloricRadiationSample operator()(double f,bool heat)const{const double d=(heat?5000.:1.)-1000.;return {1000.+f*d,1542.*f*d,1.};}};
 const auto power=[](double t){return 1e-8*(std::pow(4000.,4)-std::pow(t,4));};
 const auto a=solveCaloricParticleRadiation(Linear{},power,100.);
 const auto b=radiationMath::solveBoundedImplicitMaterialTemperature(1000.,1.,5000.,1542.,100.,100,1e-10,1e-9,power,power);
 assert(std::abs(a.temperature-b.temperatureK)<1e-6);
 std::cout<<"CALORIC_RADIATION_PASS cases="<<cases<<" heterogeneous_carry_bounds_attached_exclusion linear_limit\n";
}
