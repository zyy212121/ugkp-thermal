#include <cuda_runtime.h>
#include <cstdio>
#include <new>
#include <cstdlib>
#include <cmath>
#include <limits>
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
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
};
// Minimal state adapter: retain the production mixture caloric inversion.
// The transaction below is the unmodified production CUDA kernel.
__device__ void recoverGasPrimitiveCell(State& s,int c){
 s.Ux[c]=s.rhoUx[c]/s.rho[c];s.Uy[c]=s.rhoUy[c]/s.rho[c];s.Uz[c]=s.rhoUz[c]/s.rho[c];
 const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
 double species[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[1]};
 s.gasSpecies.cellStatus[c]=int(ugkwp::invertMixtureEnergy(species,s.rhoE[c]-kinetic,s.gasSpecies.thermo,s.gasSpecies.thermoControls,s.Tgas[c]));
}
#include "applications/gasUGKPSteady/private_backend/SteadyMixtureParticleCoupling.cuh"
struct Data {State s;double rho=1,mx=0,my=0,mz=0,energy=0,ux=0,uy=0,uz=0,T=1000,V=.7,eps=.8;
 double species[2]={.9,.1},coeff[6]={1000,-.22,0,1000,-.22,0};ugkwp::SpeciesThermoData<double> metadata[2];
 int count=1,status=1,cell=0,offsets[2]={0,1},index=0,thermoStatus=0;
 double mass=1e-6,diameter=.001,vx=0,vy=0,vz=0,oldx=0,oldy=0,oldz=0,theta=0,Tp=10000;
};
int main(int argc,char** argv){Data* d=nullptr;if(cudaMallocManaged(&d,sizeof(Data))!=cudaSuccess)return 2;new(d)Data{};auto&s=d->s; if(argc>2)d->coeff[1]=d->coeff[4]=std::strtod(argv[2],nullptr); if(argc>3)d->Tp=std::strtod(argv[3],nullptr); const double initialTp=d->Tp;
 s.nCells=1;s.rho=&d->rho;s.rhoUx=&d->mx;s.rhoUy=&d->my;s.rhoUz=&d->mz;s.rhoE=&d->energy;
 s.Ux=&d->ux;s.Uy=&d->uy;s.Uz=&d->uz;s.Tgas=&d->T;s.V=&d->V;s.epsGPrev=&d->eps;
 s.gasMu=1.8e-5;s.gasCp=3.5;s.gasPrClamped=.71;s.gasThermalConductivity=.06;
 s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.rho=d->species;s.gasSpecies.cellStatus=&d->thermoStatus;
 s.gasSpecies.thermo.species=d->metadata;s.gasSpecies.thermo.coefficients=d->coeff;s.gasSpecies.thermo.coefficientCount=6;
 s.gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;s.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;
 for(int k=0;k<2;++k){auto&m=d->metadata[k];m.molarMass=.028;m.minTemperature=100;m.maxTemperature=3000;m.coefficientOffset=3*k;}
 d->energy=ugkwp::mixtureEnergy(d->species,d->T,s.gasSpecies.thermo);
 s.particleCountDevice=&d->count;s.pStatus=&d->status;s.pCellId=&d->cell;s.cellParticleOffset=d->offsets;s.sortedParticleIndex=&d->index;
 s.pm=&d->mass;s.pd=&d->diameter;s.pux=&d->vx;s.puy=&d->vy;s.puz=&d->vz;s.puxOld=&d->oldx;s.puyOld=&d->oldy;s.puzOld=&d->oldz;s.pTheta=&d->theta;s.pT=&d->Tp;
 const double volume=d->eps*d->V,oldEnergy=d->energy,capacity=d->mass*800,dt=argc>1?std::strtod(argv[1],nullptr):100;
 const double a=-expm1(-6*2*.06/(2e6*.001*.001)*dt);
 // Independent long-double analytic solution for the linear-Cp fixture.
 const long double cv=1000.L+(long double)d->coeff[1]*1000.L
     -(long double)ugkwp::universalGasConstant<double>()/.028L;
 const long double linear=cv+(long double)capacity*a/volume;
 const long double rhs=(long double)capacity*a/volume*(initialTp-1000.L);
 const long double delta=2*rhs/(linear+sqrtl(linear*linear+2*d->coeff[1]*rhs));
 const double target=double(1000.L+delta),expected=initialTp+a*(target-initialTp);
 steadyMixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag><<<1,32>>>(&d->s,dt,false);
 const auto error=cudaDeviceSynchronize();if(error!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(error));return 2;}
 printf("expected gas target %.17g particle %.17g; actual gas %.17g particle %.17g status %d; minimum valid Cv %.17g\n",target,expected,d->T,d->Tp,d->thermoStatus,1000+d->coeff[1]*3000-ugkwp::universalGasConstant<double>()/.028);
 const long double gasHeat=(long double)volume*((long double)d->energy-oldEnergy);
 const long double parcelHeat=(long double)capacity*((long double)d->Tp-initialTp);
 const long double equation=(long double)d->Tp-initialTp-(long double)a*((long double)d->T-initialTp);
 const long double gasDelta=(long double)d->T-1000.L;
 const long double caloricHeat=(long double)volume*(cv*gasDelta+.5L*d->coeff[1]*gasDelta*gasDelta);
 const long double caloricResidual=caloricHeat+(long double)capacity*a*((long double)d->T-initialTp);
 const long double budgetTolerance=16*std::numeric_limits<double>::epsilon()*fabs(volume*oldEnergy);
 printf("parcel equation residual %.17Lg K; caloric equation residual %.17Lg J; gas heat %.17Lg J; parcel heat %.17Lg J; budget residual %.17Lg J; roundoff budget %.17Lg J\n",
        equation,caloricResidual,gasHeat,parcelHeat,gasHeat+parcelHeat,budgetTolerance);
 const bool correct=d->thermoStatus==0 && d->T>=100 && d->T<=3000
     && fabs(d->Tp-expected)<1e-8 && fabs(d->T-target)<2e-9
     && fabsl(equation)<1e-8 && fabsl(caloricResidual)<budgetTolerance
     && fabsl(gasHeat+parcelHeat)<budgetTolerance && fabsl(parcelHeat)>0
     && d->species[0]==.9 && d->species[1]==.1 && d->rho==1;
 cudaFree(d);return correct?0:1;
}
