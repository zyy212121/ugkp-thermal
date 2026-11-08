#include <cuda_runtime.h>
#include "ParticleCaloricHeatResponse.H"
#include "gasTransport/MixtureHeatExchange.H"
#include "gasTransport/MixtureParticleDirectory.cuh"
#include "gasTransport/MixtureCouplingTeam.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
constexpr int N=257;
struct State {
 int particleCapacity=N,active=N;int* particleCountDevice;
 int pStatus[N],pCellId[N],pStuck[N],cellParticleOffset[2],preBaseCellOffset[2],sortedParticleIndex[N];
 double pm[N];ugkwp::ParticleCaloricHeatResponse parcels[N];
};
namespace mixtureThermalParticles {
template<class S> __device__ bool member(const S&s,int i,int c){return mixtureParticles::member(s,i,c);}
template<class S> __device__ auto cachedResponse(const S&s,int i){return s.parcels[i];}
#include "GpuParticleCaloricResponseSum.cuh"
}
template<class Team> __device__ int test(const State& s,const Team& team,bool csr,bool split){
 const mixtureThermalParticles::Response<State,Team> response{s,0,1.,1.,csr,split,team};
 int failures=0;
 // Repeated nearby evaluations reuse a segment; large jumps cross caloric and
 // phase cuts. Direct per-parcel calorics remain the independent oracle.
 for(int j=0;j<100;++j){
  const double t=j%10==0?100.:j%10==1?3000.:1000.+(j%10)*.03;
  double q,d;const auto status=response(t,q,d);
  double exact=0.,derivative=0.,scale=0.;
  for(int i=team.rank();i<N;i+=team.width())if(s.pStatus[i]==1&&s.pStuck[i]==0){
   double h,dh;s.parcels[i](t,h,dh);exact+=s.pm[i]*h;derivative+=s.pm[i]*dh;scale+=fabs(s.pm[i]*h);
  }
  exact=team.sum(exact);derivative=team.sum(derivative);scale=team.sum(scale);
  if(status!=ugkwp::ThermoStatus::Success||fabs(q-exact)>2e-11*fmax(1.,scale)
     ||fabs(d-derivative)>2e-12*fmax(1.,fabs(derivative)))++failures;
 }
 ugkwp::SpeciesThermoData<double> metadata[2];double coefficients[]={1000.,-.22,0.,1100.,-.1,0.};
 for(int k=0;k<2;++k){metadata[k].molarMass=.028;metadata[k].minTemperature=100.;metadata[k].maxTemperature=3000.;metadata[k].coefficientOffset=3*k;}
 ugkwp::SpeciesThermoView<double,2> view{metadata,coefficients,6};
 double rho[]={.9,.1};const double oldEnergy=ugkwp::mixtureEnergy(rho,1000.,view);
 ugkwp::ThermoInversionControls<double> controls;controls.relativeEnergyTolerance=1e-13;controls.relativeTemperatureTolerance=1e-12;
 double temperature=1000.;
 const auto status=ugkwp::invertMixtureParticleHeatExchange(rho,oldEnergy,response,view,controls,temperature);
 double heat=0.;
 for(int i=team.rank();i<N;i+=team.width())if(s.pStatus[i]==1&&s.pStuck[i]==0){
  double h,dh;s.parcels[i](temperature,h,dh);heat+=s.pm[i]*h;
 }
 heat=team.sum(heat);
 const double residual=ugkwp::mixtureEnergy(rho,temperature,view)-oldEnergy+heat;
 if(status!=ugkwp::ThermoStatus::Success||temperature<100.||temperature>3000.
   ||fabs(residual)>2e-8*fmax(1.,fabs(heat)))++failures;
 return failures;
}
__global__ void checkResponse(State* s,int* errors,bool cooperative,bool csr,bool split){
 if(cooperative){const int result=test(*s,mixtureParticles::WarpCouplingTeam{},csr,split);if(!threadIdx.x)*errors=result;}
 else if(!threadIdx.x)*errors=test(*s,mixtureParticles::SerialCouplingTeam{},csr,split);
}
void checked(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
int main(){
 State* s;int* errors;checked(cudaMallocManaged(&s,sizeof(State)));checked(cudaMallocManaged(&errors,sizeof(int)));
 *s=State{};s->particleCountDevice=&s->active;
 for(int mode=0;mode<4;++mode)for(bool cooperative:{false,true})for(bool csr:{false,true})for(bool split:{false,true}){
  s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=split?N-N/2:N;
  s->preBaseCellOffset[0]=0;s->preBaseCellOffset[1]=split?N/2:0;
  for(int i=0;i<N;++i){
   s->pStatus[i]=i%17?1:0;s->pCellId[i]=0;s->pStuck[i]=i%13?0:2;s->pm[i]=1e-7*(1+i%5);
   s->sortedParticleIndex[i]=split?N/2+i:N-1-i;
   ugkwp::ParticleCaloric law;law.phaseChange=mode%2;
   law.solidCp=900.;law.latentHeat=300000.;law.mushyRange=20.;
   const double old=mode==2?10000.:500.+17*i;
   const double fraction=mode==3?1e-14:(i%7==0?0.:.02+.003*(i%100));
   s->parcels[i]={law,old,fraction,100.,10000.};
  }
  checkResponse<<<1,32>>>(s,errors,cooperative,csr,split);checked(cudaGetLastError());checked(cudaDeviceSynchronize());
  if(*errors){std::printf("FAIL mode=%d warp=%d csr=%d split=%d errors=%d\n",mode,cooperative,csr,split,*errors);return 1;}
 }
 checked(cudaFree(errors));checked(cudaFree(s));
 std::puts("PASS 32 serial/warp/directory caloric aggregate and gas-root cases; original per-parcel equation and paired energy checked");
}
