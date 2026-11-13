// Checkpoint 2A: production-header mixed coupling energy regression.
// Numerical execution belongs to the parent checkpoint batch.
// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
using std::sqrt;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "GpuOperatorContract.cuh"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuLesAlgebra.cuh"
using Real=GpuReal;
using Time=double;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME Time
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_R(1e-30)
#define __host__
#define __global__
#define __device__
#define __forceinline__ inline
// PTX traps are represented by process failure in this serial host emulator.
#define asm(...) std::abort()
struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;
constexpr Real OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr Real OfVSmall=GPU_R(1e-30);
constexpr Real OfGreat=GPU_R(1.0)/OfSmall;
struct GasPrimDevice { Real rho,ux,uy,uz,p,T; };
#include "gasTransport/GasStateView.H"
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
#include "operators/computeSstGradientsKernel.cuh"
#include "operators/sstVelocityInvariants.cuh"
#include "operators/computeGasHllcAdcSensorKernel.cuh"
#include "operators/updateBarthLimiter.cuh"
#include "operators/computeGasGradientLimiterKernel.cuh"
#include "operators/computeGasEddyViscosityKernel.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "operators/gasFaceSubgridTransportProperties.cuh"
#include "operators/computeRiemannGasFaceFluxDevice.cuh"
#include "operators/computeGasInternalFaceFluxKernel.cuh"
#include "operators/computeGasFluxPositivityScaleKernel.cuh"
#include "operators/computeSstFaceFluxKernel.cuh"


#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

template<class T>void allocateField(T&){}
template<class T>void allocateField(T*&p){p=new T[8]{};}

#ifdef CP3_FP64
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients,double>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
#else
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients,GpuEnergy,GpuGeometry,GpuWallEnergy>{ugkwp::GasSpeciesState<Real,2,double> gasSpecies;};
#endif
void ck(bool good,const char*message){if(!good){std::fprintf(stderr,"%s\n",message);std::abort();}}
void initialise(State&s){
#define UGKWP_GAS_STATE_FIELD(Type,Name) allocateField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gasFluxScheme=1;s.gasReconstruction=0;s.gasLimiter=0;s.gasMu=0;s.gasPrClamped=.7;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.rhoMin=1e-12;s.TgasMin=1e-9;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
auto&g=s.gasSpecies;g.rho=new Real[4]{};g.initial=new Real[4]{};g.flux=new Real[6]{};g.gradX=new Real[4]{};g.gradY=new Real[4]{};g.gradZ=new Real[4]{};g.limiter=new Real[2]{};g.boundaryMassFraction=new Real[6]{};g.compositionBoundaryFixed=new int[3]{};g.positivityScale=new Real[4]{};g.soundSpeed=new Real[2]{};g.heatCapacity=new Real[2]{};g.gasConstant=new Real[2]{};g.cellStatus=new int[2]{};g.faceStatus=new int[3]{};
auto*meta=new ugkwp::SpeciesThermoData<double>[2];auto*coeff=new double[6]{1000,0,-1e6,1000,0,2e6};g.thermo.species=meta;g.thermo.coefficients=coeff;g.thermo.coefficientCount=6;g.thermoControls.relativeEnergyTolerance=Real(2e-6);g.thermoControls.relativeTemperatureTolerance=Real(2e-6);
for(int k=0;k<2;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].referencePressure=101325;meta[k].coefficientOffset=3*k;}
g.rho[0]=.9;g.rho[1]=.1;g.rho[2]=.1;g.rho[3]=.9;
for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.rho[c]=1;double rhos[2]={g.rho[c],g.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(rhos,double(700),g.thermo);s.Tgas[c]=700;s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;}
s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;}
blockDim.x=1;blockIdx.x=0;
for(int c=0;c<2;++c){threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
}

#include "gpu/GpuCouplingMath.H"
ThreadIndex gridDim;
#include "ParticleCaloricHeatResponse.H"
#include "GpuParticleThermalCarry.H"
struct PairedState:State { GpuReal thetaDragAlpha[2]={1,1};
 unsigned char pStuck[4]={};
 double pThermalEnthalpyResidual[4]={},mixtureHeatInitialTemperature[4]={},mixtureHeatFraction[4]={};
 ugkwp::ParticleCaloric law;
 GpuReal couplingDragBeta[4]{}; double couplingResponseFraction[4]{};
 int count=3; int*particleCountDevice=&count; int preBaseCellOffset[3]={0,2,3};
 int particleCapacity=4,cellParticleOffset[3]={0,2,3},sortedParticleIndex[3]={0,1,2};
 int pStatus[4]={1,1,1,0},pCellId[4]={0,0,1,0};
 GpuReal pm[4]={.2,.4,.3,.1},pd[4]={.001,.0002,.0005,.001};
 GpuReal pux[4]={-10,30,2,999},puy[4]={3,-4,1,999},puz[4]={1,2,-2,999};
 GpuReal puxOld[4]={},puyOld[4]={},puzOld[4]={},pTheta[4]={2,4,3,999},pT[4]={300,1200,900,999};
 GpuReal epsGPrev[2]={.8,.6},rhoSolid=2500,particleThermalRho=2500,particleCp=800,particleThermalCapacity=2e6;
 double dragParameter0=.8,dragParameter1=0,dragParameter2=0,dragParameter3=0;
 GpuReal gasPrOneThird=std::cbrt(.71),TpMin=100,TpMax=2000;
 double gravityX=.2,gravityY=-.3,gravityZ=.4;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
};
#include "gasTransport/MixtureParticleCoupling.cuh"
ugkwp::ParticleCaloric particleCaloricDevice(const PairedState&s,int){return s.law;}
double particleEffectiveTemperatureDevice(const PairedState&s,int i){
 return ugkwp::effectiveParticleTemperature(ugkwp::ParticleThermalCarry<GpuReal>{s.pT[i],s.pThermalEnthalpyResidual[i]},s.law);
}
#define DeviceState PairedState
#include "GpuParticleGasHeatRelaxation.cuh"
#undef DeviceState
#include "GpuMixtureThermalParticleHeat.cuh"

void setup(PairedState&s){initialise(s);gridDim.x=1;s.gasMu=1.8e-5;s.gasPrClamped=.71;s.gasThermalConductivity=.06;s.V[0]=.7;s.V[1]=1.3;
for(int c=0;c<2;++c){s.rhoUx[c]=s.rho[c]*5;s.rhoUy[c]=s.rho[c]*-2;s.rhoUz[c]=s.rho[c]*1;s.rhoE[c]+=.5*s.rho[c]*(25+4+1);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}}
void closeValue(long double a,long double b,long double tol,const char*why){if(fabsl(a-b)>tol){std::fprintf(stderr,"%s: %.21Lg != %.21Lg difference %.21Lg\n",why,a,b,a-b);std::abort();}}
struct Budget {long double mass=0,x=0,y=0,z=0,energy=0;};
Budget budget(const PairedState&s){Budget b;
for(int c=0;c<2;++c){long double v=(long double)s.epsGPrev[c]*s.V[c];b.mass+=v*s.rho[c];b.x+=v*s.rhoUx[c];b.y+=v*s.rhoUy[c];b.z+=v*s.rhoUz[c];b.energy+=v*s.rhoE[c];}
for(int i=0;i<s.particleCapacity;++i)if(s.pStatus[i]==1){long double m=s.pm[i];b.mass+=m;b.x+=m*s.pux[i];b.y+=m*s.puy[i];b.z+=m*s.puz[i];b.energy+=m*(.5L*((long double)s.pux[i]*s.pux[i]+(long double)s.puy[i]*s.puy[i]+(long double)s.puz[i]*s.puz[i])+1.5L*s.pTheta[i]+800.L*s.pT[i]);}return b;}
void conserved(const Budget&a,const Budget&b){
 const long double mt=sizeof(GpuReal)==4?2e-5L:1e-14L,pt=sizeof(GpuReal)==4?2e-4L:1e-12L,et=sizeof(GpuReal)==4?2.L:5e-8L;
 closeValue(a.mass,b.mass,mt,"mass");closeValue(a.x,b.x,pt,"momentum x");closeValue(a.y,b.y,pt,"momentum y");closeValue(a.z,b.z,pt,"momentum z");closeValue(a.energy,b.energy,et,"energy");
}

#include <chrono>
#include <sstream>
#include <iostream>
#include <memory>
#include <vector>
volatile double resultSink=0; double times[5]={}; double measurementT=0,measurementP=0; std::ostringstream measures; int caseId=0;

void record(const PairedState&s,double error){
 const int id=caseId++;
 auto value=[&](const std::string&key,double v){if(measures.tellp()>0)measures<<",";measures<<"\"case"<<id<<"_"<<key<<"\":"<<v;};
 for(int c=0;c<2;++c){value("Tgas"+std::to_string(c),s.Tgas[c]);value("rhoE"+std::to_string(c),s.rhoE[c]);}
 for(int i=0;i<3;++i){value("pT"+std::to_string(i),s.pT[i]);value("pux"+std::to_string(i),s.pux[i]);}
 value("budget_error",error);
}
void directory(PairedState&s,int mode){if(mode==2){
 // Same active particles: bulk cell0={0}; injected cell0={1}, cell1={2}.
 s.preBaseCellOffset[0]=0;s.preBaseCellOffset[1]=1;s.preBaseCellOffset[2]=1;
 s.cellParticleOffset[0]=0;s.cellParticleOffset[1]=1;s.cellParticleOffset[2]=2;
 s.sortedParticleIndex[0]=1;s.sortedParticleIndex[1]=2;
}}
template<class Model>void step(PairedState&s,double dt,int mode){
 for(int c=0;c<2;++c)mixtureParticleCouplingCell<Model,true>(s,c,dt,true,mode==1,mode==2,mixtureParticles::SerialCouplingTeam{});
}
template<class Model>void check(int mode) {
 PairedState s;setup(s);directory(s,mode);const auto before=budget(s);
 step<Model>(s,0,mode);conserved(before,budget(s));
 ck(s.pT[0]==300&&s.pT[1]==1200&&s.pux[0]==-10,"dt0");
 for(int k=0;k<2;++k){
  for(int i=0;i<3;++i){s.couplingDragBeta[i]=-123;s.couplingResponseFraction[i]=-456;}
  step<Model>(s,k?.07:.03,mode);conserved(before,budget(s));
  for(int c=0;c<2;++c)ck(s.gasSpecies.cellStatus[c]==0&&std::isfinite(s.Tgas[c]),"EOS convergence/finite");
  for(int i=0;i<3;++i)ck(std::isfinite(s.pT[i])&&std::isfinite(s.pux[i]),"particle finite");
  ck(s.pT[0]>300&&s.pT[1]<1200,"hot/cold trend");
  ck(s.pT[3]==999&&s.pux[3]==999,"inactive slot");
  measurementT+=s.Tgas[0];measurementP+=s.pT[0];record(s,double(budget(s).energy-before.energy));
  std::cout<<"mode="<<mode<<" call="<<k<<" energy_error="<<double(budget(s).energy-before.energy)<<" Tg="<<s.Tgas[0]<<" Tp="<<s.pT[0]<<std::endl;
 }
}
template<class Model>void perf(int mode){
 constexpr int n=128;
 for(int sample=-1;sample<5;++sample){
  std::vector<std::unique_ptr<PairedState>> batch;
  for(int i=0;i<n;++i){batch.emplace_back(new PairedState);setup(*batch.back());directory(*batch.back(),mode);}
  auto start=std::chrono::steady_clock::now();
  for(auto&s:batch)step<Model>(*s,.03,mode);
  auto stop=std::chrono::steady_clock::now();
  for(auto&s:batch)resultSink+=s->rhoE[0]+s->pT[0];
  if(sample>=0)times[sample]+=std::chrono::duration<double>(stop-start).count();
  if(sample>=0)std::cout<<"sample="<<sample<<" full_steps="<<n<<" cells_per_step=2 particles_per_step=3 mode="<<mode<<" ns="<<std::chrono::duration<double,std::nano>(stop-start).count()<<" allocated_scratch_bytes="<<4*(sizeof(GpuReal)+sizeof(double))<<std::endl;
 }
}

double nonlinearEnergy(const PairedState&s){
 double e=0;for(int c=0;c<2;++c)e+=double(s.epsGPrev[c])*s.V[c]*s.rhoE[c];
 for(int i=0;i<3;++i)e+=double(s.pm[i])*(s.law.enthalpy(s.pT[i])+s.pThermalEnthalpyResidual[i]+.5*(double(s.pux[i])*s.pux[i]+double(s.puy[i])*s.puy[i]+double(s.puz[i])*s.puz[i])+1.5*s.pTheta[i]);
 return e;
}
template<bool Observe=false> void nonlinearStep(PairedState&s,double dt,int mode){
 double preT[2]={},expected[3]={},stale[3]={};
 if constexpr(Observe){preT[0]=s.Tgas[0];preT[1]=s.Tgas[1];}
 auto observable=[&](const std::string&key,double value){
 if(measures.tellp()>0)measures<<",";
 measures<<"\"case"<<caseId<<"_"<<key<<"\":"<<value;
 };

 for(int c=0;c<2;++c)mixtureParticleCouplingCell<ugkwpGpuDrag::SchillerNaumannDeviceDrag,false>(s,c,dt,true,mode==1,mode==2,mixtureParticles::SerialCouplingTeam{});

 if constexpr(Observe){
  bool sensitive=false;
  for(int c=0;c<2;++c){
   const auto postT=s.Tgas[c];
   observable("preT"+std::to_string(c),preT[c]);observable("postT"+std::to_string(c),postT);
   observable("preMu"+std::to_string(c),ugkwp::gasCellMolecularViscosityAt(s,c,GpuReal(preT[c])));
   observable("postMu"+std::to_string(c),ugkwp::gasCellMolecularViscosityAt(s,c,postT));
   observable("preK"+std::to_string(c),molecularGasConductivity(s,c,GpuReal(preT[c])));
   observable("postK"+std::to_string(c),molecularGasConductivity(s,c,postT));
   for(int i=0;i<3;++i)if(s.pCellId[i]==c){
    expected[i]=mixtureThermalParticles::response(s,c,i,dt).fraction;
    s.Tgas[c]=GpuReal(preT[c]);stale[i]=mixtureThermalParticles::response(s,c,i,dt).fraction;s.Tgas[c]=postT;
    sensitive=sensitive||std::abs(expected[i]-stale[i])>1e-10;
   }
  }
  ck(sensitive,"temperature-sensitive source epoch not exercised");
 }
#ifndef CP3_CELL_PREPARES_HEAT
 threadIdx.x=0;prepareMixtureThermalParticleHeatKernel(&s,dt);
#endif
 for(int c=0;c<2;++c)mixtureThermalParticleHeatCell(s,c,dt,mode==1,mode==2,mixtureParticles::SerialCouplingTeam{});
 if constexpr(Observe)for(int i=0;i<3;++i){
  ck(std::abs(s.mixtureHeatFraction[i]-expected[i])<=1e-10,"heat cache used wrong source temperature");
  observable("expectedFraction"+std::to_string(i),expected[i]);observable("staleFraction"+std::to_string(i),stale[i]);observable("actualFraction"+std::to_string(i),s.mixtureHeatFraction[i]);
 }

}
void nonlinearCheck(int mode,int crossing=0){
 PairedState s;setup(s);directory(s,mode);s.law.phaseChange=true;s.law.solidCp=1000;
 if(crossing){
 s.TpMax=5000; s.pT[0]=4100;s.pT[1]=2100;s.pT[2]=3900;
 for(int k=0;k<2;++k)const_cast<ugkwp::SpeciesThermoData<double>*>(s.gasSpecies.thermo.species)[k].maxTemperature=5000;
 for(int c=0;c<2;++c){double mass[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
 s.rhoE[c]=ugkwp::mixtureEnergy(mass,crossing==1?4500.:300.,s.gasSpecies.thermo)+.5*(double(s.rhoUx[c])*s.rhoUx[c]+double(s.rhoUy[c])*s.rhoUy[c]+double(s.rhoUz[c])*s.rhoUz[c])/s.rho[c];recoverGasPrimitiveCell(s,c);}
 }
 if(!crossing){
 s.gasViscosity.model=ugkwp::MolecularViscosityModel::MassFractionSutherland;
 s.gasViscosity.compositionSpecies=0;s.gasViscosity.compositionTRef=300;
 s.gasViscosity.muCoefficients[0]=1.8e-5;s.gasViscosity.muCoefficients[1]=s.gasViscosity.muCoefficients[2]=0;
 s.gasViscosity.sCoefficients[0]=110.4;s.gasViscosity.sCoefficients[1]=s.gasViscosity.sCoefficients[2]=0;
 s.gasThermalConductivity=-1;s.gasThermalConductivityConfigured=false;
 s.pux[0]=-200;s.pux[1]=300;s.pux[2]=20;
 }
 const double e=nonlinearEnergy(s);
 for(int repeat=0;repeat<2;++repeat){
  for(int i=0;i<3;++i){s.mixtureHeatInitialTemperature[i]=-999;s.mixtureHeatFraction[i]=-999;}
  if(crossing)nonlinearStep(s,3.,mode);else nonlinearStep<true>(s,repeat?.07:.03,mode);
  for(int c=0;c<2;++c)ck(s.gasSpecies.cellStatus[c]==0&&std::isfinite(s.Tgas[c]),"nonlinear EOS");
  // Exercise the actual mutable aggregate cache, not only a bare parcel law.
  for(int c=0;c<2;++c){
   const double vol=double(s.epsGPrev[c])*s.V[c];
   const mixtureThermalParticles::Response<PairedState,mixtureParticles::SerialCouplingTeam> reused{s,c,.03,vol,mode==1,mode==2,{}};
   const double beforeT=s.pT[0],beforeE=s.rhoE[c];
   for(double trial:{2327.,2327.001,4000.,3999.999,100.,5000.,2600.}){
    double q,dq,referenceQ,referenceD;
    const mixtureThermalParticles::Response<PairedState,mixtureParticles::SerialCouplingTeam> fresh{s,c,.03,vol,mode==1,mode==2,{}};
    ck(reused(trial,q,dq)==ugkwp::ThermoStatus::Success&&fresh(trial,referenceQ,referenceD)==ugkwp::ThermoStatus::Success,"cache evaluation");
    ck(std::abs(q-referenceQ)<1e-8*std::max(1.,std::abs(referenceQ))&&std::abs(dq-referenceD)<1e-8*std::max(1.,std::abs(referenceD)),"endpoint cache isolation");
   }
   ck(s.pT[0]==beforeT&&s.rhoE[c]==beforeE,"trial published state");
  }

  ck(std::abs(nonlinearEnergy(s)-e)<(sizeof(GpuReal)==4?2.:5e-8),"nonlinear budget");
  ck((crossing?(crossing==1?s.pT[1]>2100:s.pT[0]<4100):(s.pT[0]>300&&s.pT[1]<1200))&&s.pT[3]==999,"nonlinear hot/cold/inactive");
  measurementT+=s.Tgas[0];measurementP+=s.pT[0];record(s,nonlinearEnergy(s)-e);
 }
}

void nonlinearPerf(int mode){
 constexpr int n=128;
 for(int sample=-1;sample<5;++sample){
  std::vector<std::unique_ptr<PairedState>> batch;
  for(int i=0;i<n;++i){batch.emplace_back(new PairedState);setup(*batch.back());directory(*batch.back(),mode);batch.back()->law.phaseChange=true;batch.back()->law.solidCp=1000;}
  auto start=std::chrono::steady_clock::now();
  for(auto&s:batch)nonlinearStep(*s,.03,mode);
  auto stop=std::chrono::steady_clock::now();
  for(auto&s:batch)resultSink+=s->rhoE[0]+s->pT[0];
  if(sample>=0)times[sample]+=std::chrono::duration<double>(stop-start).count();
 }
}
int main(int argc,char**argv){
 std::cout<<std::setprecision(17);measures<<std::setprecision(17);
 bool performance=argc>1&&std::string(argv[1])=="perf";
 for(int m=0;m<3;++m)if(performance){nonlinearPerf(m);perf<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(m);perf<ugkwpGpuDrag::GidaspowErgunWenYuDeviceDrag>(m);}
 else{nonlinearCheck(m);nonlinearCheck(m,1);nonlinearCheck(m,2);check<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(m);check<ugkwpGpuDrag::GidaspowErgunWenYuDeviceDrag>(m);}
 std::cout<<"RESULT_JSON={\"family\":\"coupling\",\"precision\":"<<sizeof(GpuReal)*8<<",\"mode\":\""<<(performance?"perf":"correctness")<<"\",\"checks\":"<<caseId<<",\"measurements\":{"<<measures.str()<<"}";

 if(performance){std::cout<<",\"samples_seconds\":[";for(int i=0;i<5;++i){if(i)std::cout<<",";std::cout<<times[i];}std::cout<<"],\"iterations\":1152,\"worksize\":3,\"setup/reset_excluded\":true,\"cache_bytes\":"<<4*(sizeof(GpuReal)+sizeof(double));}
 std::cout<<"}"<<std::endl;

}
