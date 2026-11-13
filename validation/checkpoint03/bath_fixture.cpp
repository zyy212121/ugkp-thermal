
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include "GpuPrecisionTypes.H"
#include "gasTransport/MixtureHeatExchange.H"
#include "ParticleCaloricHeatResponse.H"
struct Response {
 ugkwp::ParticleCaloricHeatResponse parcel;
 ugkwp::ThermoStatus operator()(double T,double&q,double&dq)const noexcept{
  parcel(T,q,dq);q*=.2;dq*=.2;return ugkwp::ThermoStatus::Success;
 }
};
void ck(bool x){if(!x)std::abort();}
volatile double sink=0;
int main(int argc,char**argv){
 bool perf=argc>1&&std::string(argv[1])=="perf";
 ugkwp::SpeciesThermoData<double> meta[1];
 double coeff[3]={1400,.05,0},mass[1]={1};
 meta[0].coefficientOffset=0;meta[0].molarMass=.028;
 meta[0].minTemperature=100;meta[0].maxTemperature=5000;meta[0].referencePressure=101325;
 ugkwp::SpeciesThermoView<double,1> view;view.species=meta;view.coefficients=coeff;view.coefficientCount=3;
 ugkwp::ThermoInversionControls<double> controls;
 controls.relativeEnergyTolerance=1e-8;controls.relativeTemperatureTolerance=1e-8;
 double times[5]={},temperatures[6]={},residuals[6]={};
 const double Tg[6]={300,1600,2200,2400,4100,5000};
 const double Tp[6]={1600,300,2600,2100,3900,5000};
 for(int sample=perf?-1:0;sample<(perf?5:1);++sample){
  auto start=std::chrono::steady_clock::now();
  for(int repeat=0;repeat<(perf?128:1);++repeat)for(int i=0;i<6;++i){
   Response response;response.parcel.law.phaseChange=true;response.parcel.law.solidCp=1000;
   response.parcel.initialTemperature=Tp[i];response.parcel.fraction=.8;
   response.parcel.minimumTemperature=100;response.parcel.maximumTemperature=5000;
   double t=Tg[i],old=ugkwp::mixtureEnergy(mass,t,view);
   double referenceHeat,referenceDerivative;response(t,referenceHeat,referenceDerivative);
   const auto status=ugkwp::invertMixtureParticleHeatExchange(mass,old,response,view,controls,t);
   if(!perf){
    double q,dq;response(t,q,dq);
    residuals[i]=ugkwp::mixtureEnergy(mass,t,view)-old+q;
    ck(status==ugkwp::ThermoStatus::Success&&std::isfinite(t));
    ck(t>=std::fmin(Tg[i],Tp[i])-.01&&t<=std::fmax(Tg[i],Tp[i])+.01);
    const double roundoff=4*std::numeric_limits<double>::epsilon()*(std::abs(old)+std::abs(referenceHeat));
    const double allowed=4*(controls.absoluteEnergyTolerance+roundoff+controls.relativeEnergyTolerance*std::abs(referenceHeat));
    ck(std::abs(residuals[i])<=allowed);
    std::cout<<std::setprecision(17)<<"bath_case="<<i<<" requested_relative_energy="<<controls.relativeEnergyTolerance<<" requested_absolute_energy="<<controls.absoluteEnergyTolerance<<" residual_allowed="<<allowed<<" residual_actual="<<residuals[i]<<std::endl;
    temperatures[i]=t;
    // Independent trial responses, including exact caloric cuts, cannot mutate parcel state.
    const auto saved=response.parcel;
    for(double trial:{100.,2327.,4000.,5000.}){response(trial,q,dq);ck(std::isfinite(q)&&std::isfinite(dq));}
    ck(response.parcel.initialTemperature==saved.initialTemperature);
   }
   sink+=t;
  }
  auto stop=std::chrono::steady_clock::now();
  if(perf&&sample>=0)times[sample]=std::chrono::duration<double>(stop-start).count();
 }
 std::cout<<std::setprecision(17)<<"RESULT_JSON={\"family\":\"bath\",\"precision\":"<<sizeof(GpuReal)*8<<",\"mode\":\""<<(perf?"perf":"correctness")<<"\",\"checks\":6,\"measurements\":{";
 if(!perf)for(int i=0;i<6;++i){if(i)std::cout<<",";std::cout<<"\"T"<<i<<"\":"<<temperatures[i]<<",\"residual"<<i<<"\":"<<residuals[i];}
 std::cout<<"}";
 if(perf){std::cout<<",\"samples_seconds\":[";for(int i=0;i<5;++i){if(i)std::cout<<",";std::cout<<times[i];}std::cout<<"],\"iterations\":128,\"worksize\":6,\"setup/reset_excluded\":false,\"cache_bytes\":0";}
 std::cout<<"}"<<std::endl;
}
