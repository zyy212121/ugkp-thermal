#include <cuda_runtime.h>
#include <cstdio>
#include <new>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "GpuParticleCaloric.H"
#include "ParticleCaloricHeatResponse.H"
#include "GpuParticleThermalCarry.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/MixtureThermo.H"
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
#include "gasTransport/GasMolecularConductivity.cuh"
struct State:ugkwp::GasStateView<double,double,int>{
 ugkwp::GasSpeciesState<double,2> gasSpecies;
 int *particleCountDevice,*pStatus,*pCellId,*cellParticleOffset,*sortedParticleIndex,particleCapacity=1;
 double *epsGPrev,*pm,*pd,*pux,*puy,*puz,*puxOld,*puyOld,*puzOld,*pTheta,*pT;
 double rhoSolid=2500,particleThermalRho=2500,particleCp=800,particleThermalCapacity=2e6;
 double dragParameter0=0,dragParameter1=0,dragParameter2=0,dragParameter3=0;
 double gasPrOneThird=cbrt(.71),TpMin=100,TpMax=10000,gravityX=0,gravityY=0,gravityZ=0;
 unsigned char* pStuck;double *pThermalEnthalpyResidual,*mixtureHeatInitialTemperature,*mixtureHeatFraction,*thetaDragAlpha;
 ugkwp::ParticleCaloric law;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
};
// Minimal state adapter: retain the production mixture caloric inversion.
// The transaction below is the unmodified production CUDA kernel.
__device__ void recoverGasPrimitiveCell(State& s,int c){
 s.Ux[c]=s.rhoUx[c]/s.rho[c];s.Uy[c]=s.rhoUy[c]/s.rho[c];s.Uz[c]=s.rhoUz[c]/s.rho[c];
 const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
 double species[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[s.nCells+c]};
 s.gasSpecies.cellStatus[c]=int(ugkwp::invertMixtureEnergy(species,s.rhoE[c]-kinetic,s.gasSpecies.thermo,s.gasSpecies.thermoControls,s.Tgas[c]));
}
#include "common/gasTransport/MixtureParticleCoupling.cuh"

using DeviceState=State;using GpuReal=double;
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_TINY(x) double(x)
template<class T> __device__ T clampMin(T a,T b){return a<b?b:a;}
template<class T> __device__ T clampRange(T a,T b,T c){return a<b?b:a>c?c:a;}
template<class T> __device__ T finiteOr(T a,T b){return isfinite(a)?a:b;}
__device__ ugkwp::ParticleCaloric particleCaloricDevice(const State&s,int){return s.law;}
__device__ double particleEffectiveTemperatureDevice(const State&s,int i){return s.law.temperature(s.law.enthalpy(s.pT[i])+s.pThermalEnthalpyResidual[i]);}
#include "GpuParticleGasHeatRelaxation.cuh"
#include "GpuMixtureThermalParticleHeat.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
void check(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::exit(2);}}
struct Allocation{void*host;void*device;size_t bytes;};
std::vector<Allocation> storage;
template<class T>T* allocate(int n){T*p=static_cast<T*>(std::malloc(n*sizeof(T)));assert(p);storage.push_back({p,nullptr,n*sizeof(T)});for(int i=0;i<n;++i)new(p+i)T{};return p;}
template<class T>T* relocated(T* p){if(!p)return nullptr;for(auto&a:storage)if(a.host==p)return static_cast<T*>(a.device);std::abort();}
State* upload(State* h){
 for(auto&a:storage)if(!a.device){check(cudaMalloc(&a.device,a.bytes));check(cudaMemcpy(a.device,a.host,a.bytes,cudaMemcpyHostToDevice));}
 State d=*h;
#define MOVE(name) d.name=relocated(h->name)
 MOVE(rho);MOVE(rhoUx);MOVE(rhoUy);MOVE(rhoUz);MOVE(rhoE);MOVE(Ux);MOVE(Uy);MOVE(Uz);MOVE(Tgas);MOVE(V);MOVE(epsGPrev);MOVE(thetaDragAlpha);
 MOVE(gasSpecies.rho);MOVE(gasSpecies.cellStatus);MOVE(gasSpecies.thermo.species);MOVE(gasSpecies.thermo.coefficients);
 MOVE(particleCountDevice);MOVE(pStatus);MOVE(pCellId);MOVE(sortedParticleIndex);MOVE(cellParticleOffset);MOVE(pStuck);
 MOVE(pm);MOVE(pd);MOVE(pux);MOVE(puy);MOVE(puz);MOVE(puxOld);MOVE(puyOld);MOVE(puzOld);MOVE(pTheta);MOVE(pT);MOVE(pThermalEnthalpyResidual);MOVE(mixtureHeatInitialTemperature);MOVE(mixtureHeatFraction);
#undef MOVE
 State* device=relocated(h);check(cudaMemcpy(device,&d,sizeof(d),cudaMemcpyHostToDevice));return device;
}
void download(State*a,State*b){for(auto&q:storage)if(q.host!=a&&q.host!=b)check(cudaMemcpy(q.host,q.device,q.bytes,cudaMemcpyDeviceToHost));}

State* create(int cells,int perCell){
 State*s=allocate<State>(1);s->nCells=cells;s->particleCapacity=cells*perCell;
#define CELL(name) s->name=allocate<double>(cells)
 CELL(rho);CELL(rhoUx);CELL(rhoUy);CELL(rhoUz);CELL(rhoE);CELL(Ux);CELL(Uy);CELL(Uz);CELL(Tgas);CELL(V);CELL(epsGPrev);CELL(thetaDragAlpha);
#undef CELL
 s->gasMu=1.8e-5;s->gasPrClamped=.71;s->gasThermalConductivity=.06;
 s->gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s->gasSpecies.rho=allocate<double>(2*cells);s->gasSpecies.cellStatus=allocate<int>(cells);
 auto*meta=allocate<ugkwp::SpeciesThermoData<double>>(2);auto*coeff=allocate<double>(6);
 s->gasSpecies.thermo.species=meta;s->gasSpecies.thermo.coefficients=coeff;s->gasSpecies.thermo.coefficientCount=6;
 s->gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;s->gasSpecies.thermoControls.relativeTemperatureTolerance=1e-13;
 for(int k=0;k<2;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].coefficientOffset=3*k;coeff[3*k]=1000;coeff[3*k+1]=-.22;coeff[3*k+2]=-1e7;}
 s->law.phaseChange=true;s->law.solidCp=900;s->law.latentHeat=8e5;
 for(int c=0;c<cells;++c){s->rho[c]=1;s->Ux[c]=s->rhoUx[c]=30;s->Tgas[c]=1000;s->V[c]=1;s->epsGPrev[c]=.8;s->thetaDragAlpha[c]=1;
  double partial[2]={.9,.1};s->gasSpecies.rho[c]=.9;s->gasSpecies.rho[cells+c]=.1;s->rhoE[c]=ugkwp::mixtureEnergy(partial,1000.,s->gasSpecies.thermo)+450;}
 const int n=cells*perCell;s->particleCountDevice=allocate<int>(1);*s->particleCountDevice=n;
 s->pStatus=allocate<int>(n);s->pCellId=allocate<int>(n);s->sortedParticleIndex=allocate<int>(n);s->cellParticleOffset=allocate<int>(cells+1);s->pStuck=allocate<unsigned char>(n);
#define PART(name) s->name=allocate<double>(n)
 PART(pm);PART(pd);PART(pux);PART(puy);PART(puz);PART(puxOld);PART(puyOld);PART(puzOld);PART(pTheta);PART(pT);PART(pThermalEnthalpyResidual);PART(mixtureHeatInitialTemperature);PART(mixtureHeatFraction);
#undef PART
 for(int c=0;c<=cells;++c)s->cellParticleOffset[c]=c*perCell;
 for(int i=0;i<n;++i){s->pStatus[i]=1;s->pCellId[i]=i/perCell;s->sortedParticleIndex[i]=i;s->pm[i]=.002/perCell;s->pd[i]=.001;s->pux[i]=i%17-8;s->puy[i]=i%11-5;s->pTheta[i]=2;s->pT[i]=2400+i%700;s->pThermalEnthalpyResidual[i]=1e-8;}
 return s;
}
template<bool Coop>void advance(State*s,bool nonlinear,int cells,int n){
 if(nonlinear){
  if constexpr(Coop)check(launchMixtureParticleCouplingCells<ugkwpGpuDrag::SchillerNaumannDeviceDrag,false>(s,cells,.0001,true,true,false));
  else mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag,false><<<(cells+127)/128,128>>>(s,.0001,true);
  prepareMixtureThermalParticleHeatKernel<<<(n+127)/128,128>>>(s,.0001);
  if constexpr(Coop)check(launchMixtureThermalParticleHeatCells(s,cells,.0001,true,false));
  else mixtureThermalParticleHeatKernel<<<(cells+127)/128,128>>>(s,.0001);
 }else {
  if constexpr(Coop)check(launchMixtureParticleCouplingCells<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(s,cells,.0001,true,true,false));
  else mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag><<<(cells+127)/128,128>>>(s,.0001,true);
 }
 check(cudaGetLastError());
}
long double total(State*s,bool nonlinear){long double e=0;for(int c=0;c<s->nCells;++c)e+=(long double)s->epsGPrev[c]*s->V[c]*s->rhoE[c];
 for(int i=0;i<s->particleCapacity;++i){long double h=nonlinear?s->law.enthalpy(s->pT[i])+s->pThermalEnthalpyResidual[i]:800*s->pT[i];
  e+=(long double)s->pm[i]*(h+.5L*(s->pux[i]*s->pux[i]+s->puy[i]*s->puy[i]+s->puz[i]*s->puz[i])+1.5L*s->pTheta[i]);}return e;}
int main(int argc,char**argv){
 int cells=argc>1?atoi(argv[1]):1,perCell=argc>2?atoi(argv[2]):8192;const int repeats=argc>3?atoi(argv[3]):20;
 for(bool nonlinear:{false,true}){
  State*a=create(cells,perCell),*b=create(cells,perCell);const long double old=total(a,nonlinear);
  State* da=upload(a);State* db=upload(b);
  advance<false>(da,nonlinear,cells,cells*perCell);check(cudaDeviceSynchronize());advance<true>(db,nonlinear,cells,cells*perCell);check(cudaDeviceSynchronize());
  float milliseconds[2]{};cudaEvent_t first,last;check(cudaEventCreate(&first));check(cudaEventCreate(&last));
  for(int mode=0;mode<2;++mode){check(cudaEventRecord(first));for(int i=0;i<repeats;++i){if(mode)advance<true>(db,nonlinear,cells,cells*perCell);else advance<false>(da,nonlinear,cells,cells*perCell);}check(cudaEventRecord(last));check(cudaEventSynchronize(last));check(cudaEventElapsedTime(&milliseconds[mode],first,last));}
  download(a,b);double error=0;for(int c=0;c<cells;++c){assert(a->gasSpecies.cellStatus[c]==0&&b->gasSpecies.cellStatus[c]==0);error=std::max(error,std::abs(a->Tgas[c]-b->Tgas[c]));error=std::max(error,std::abs(a->rhoE[c]-b->rhoE[c])/1000);}
  for(int i=0;i<cells*perCell;++i){error=std::max(error,std::abs(a->pT[i]-b->pT[i]));error=std::max(error,std::abs(a->pux[i]-b->pux[i]));}
  const long double budget=std::max(fabsl(total(a,nonlinear)-old),fabsl(total(b,nonlinear)-old));
  assert(error<2e-7&&budget<5e-10L*std::max(1.L,fabsl(old)));
  std::cout<<std::setprecision(17)<<"PASS cells="<<cells<<" parcels_per_cell="<<perCell<<" nonlinear="<<nonlinear<<" steps="<<repeats+1<<" scalar_ms="<<milliseconds[0]/repeats<<" cooperative_ms="<<milliseconds[1]/repeats<<" maximum_difference="<<error<<" energy_budget="<<double(budget)<<'\n';
  check(cudaEventDestroy(first));check(cudaEventDestroy(last));for(auto&p:storage){check(cudaFree(p.device));std::free(p.host);}storage.clear();
 }
}
