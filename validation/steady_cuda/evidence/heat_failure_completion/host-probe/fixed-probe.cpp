// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
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
#include "legacy_gas_fixture.hpp"
#ifdef LEGACY_REFERENCE
using DeviceState=LegacyGasFixture;
#else
// No application DeviceState definition: the actual operators must be generic.
#include "gasTransport/GasStateView.H"
#endif
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

#ifdef LEGACY_OPTIONAL_SPECIES
struct TestLegacyGasFixture:LegacyGasFixture{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
#else
using TestLegacyGasFixture=LegacyGasFixture;
#endif

struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
void ck(bool good,const char*message){if(!good){std::fprintf(stderr,"%s\n",message);std::abort();}}
void initialise(State&s){
s.diagnostics=new ugkp::diagnostics::DeviceContext[8]{};
s.Cx=new Real[8]{};
s.Cy=new Real[8]{};
s.Cz=new Real[8]{};
s.Sfx=new Real[8]{};
s.Sfy=new Real[8]{};
s.Sfz=new Real[8]{};
s.Tgas=new Real[8]{};
s.Ux=new Real[8]{};
s.Uy=new Real[8]{};
s.Uz=new Real[8]{};
s.V=new Real[8]{};
s.cellFaceId=new int[8]{};
s.cellLength=new Real[8]{};
s.cellPlaneCount=new int[8]{};
s.cellPlaneStart=new int[8]{};
s.deltaCoeffs=new Real[8]{};
s.faceCx=new Real[8]{};
s.faceCy=new Real[8]{};
s.faceCz=new Real[8]{};
s.faceNeighbour=new int[8]{};
s.faceOwner=new int[8]{};
s.facePeriodicDx=new Real[8]{};
s.facePeriodicDy=new Real[8]{};
s.facePeriodicDz=new Real[8]{};
s.facePeriodicPair=new int[8]{};
s.faceWeight=new Real[8]{};
s.gasBoundaryKind=new int[8]{};
s.gasBoundaryP=new Real[8]{};
s.gasBoundaryPFix=new int[8]{};
s.gasBoundaryPWave=new int[8]{};
s.gasBoundaryPWaveFieldInf=new Real[8]{};
s.gasBoundaryPWaveGamma=new Real[8]{};
s.gasBoundaryPWaveLInf=new Real[8]{};
s.gasBoundaryRho=new Real[8]{};
s.gasBoundaryRhoFix=new int[8]{};
s.gasBoundaryT=new Real[8]{};
s.gasBoundaryTFix=new int[8]{};
s.gasBoundaryUFix=new int[8]{};
s.gasBoundaryUx=new Real[8]{};
s.gasBoundaryUy=new Real[8]{};
s.gasBoundaryUz=new Real[8]{};
s.gasFluxPositivityScale=new Real[8]{};
s.gasGradientLimiterP=new Real[8]{};
s.gasGradientLimiterRho=new Real[8]{};
s.gasGradientLimiterT=new Real[8]{};
s.gasGradientLimiterUx=new Real[8]{};
s.gasGradientLimiterUy=new Real[8]{};
s.gasGradientLimiterUz=new Real[8]{};
s.gasHllcAdcSensor=new Real[8]{};
s.gasPhiRho=new Real[8]{};
s.gasPhiRhoE=new Real[8]{};
s.gasPhiRhoUx=new Real[8]{};
s.gasPhiRhoUy=new Real[8]{};
s.gasPhiRhoUz=new Real[8]{};
s.gradKX=new Real[8]{};
s.gradKY=new Real[8]{};
s.gradKZ=new Real[8]{};
s.gradOmegaX=new Real[8]{};
s.gradOmegaY=new Real[8]{};
s.gradOmegaZ=new Real[8]{};
s.gradPx=new Real[8]{};
s.gradPy=new Real[8]{};
s.gradPz=new Real[8]{};
s.gradRhoX=new Real[8]{};
s.gradRhoY=new Real[8]{};
s.gradRhoZ=new Real[8]{};
s.gradTX=new Real[8]{};
s.gradTY=new Real[8]{};
s.gradTZ=new Real[8]{};
s.gradUxX=new Real[8]{};
s.gradUxY=new Real[8]{};
s.gradUxZ=new Real[8]{};
s.gradUyX=new Real[8]{};
s.gradUyY=new Real[8]{};
s.gradUyZ=new Real[8]{};
s.gradUzX=new Real[8]{};
s.gradUzY=new Real[8]{};
s.gradUzZ=new Real[8]{};
s.k=new Real[8]{};
s.magSf=new Real[8]{};
s.nut=new Real[8]{};
s.omega=new Real[8]{};
s.p=new Real[8]{};
s.pressureScheduleTimes=new Time[8]{};
s.pressureScheduleValues=new Real[8]{};
s.rho=new Real[8]{};
s.rhoE=new Real[8]{};
s.rhoENext=new Real[8]{};
s.rhoK=new Real[8]{};
s.rhoKInitial=new Real[8]{};
s.rhoNext=new Real[8]{};
s.rhoOmega=new Real[8]{};
s.rhoOmegaInitial=new Real[8]{};
s.rhoUx=new Real[8]{};
s.rhoUxNext=new Real[8]{};
s.rhoUy=new Real[8]{};
s.rhoUyNext=new Real[8]{};
s.rhoUz=new Real[8]{};
s.rhoUzNext=new Real[8]{};
s.riemannBoundaryKind=new int[8]{};
s.riemannBoundaryP=new Real[8]{};
s.riemannBoundaryPFix=new int[8]{};
s.riemannBoundaryPWave=new int[8]{};
s.riemannBoundaryRho=new Real[8]{};
s.riemannBoundaryRhoFix=new int[8]{};
s.riemannBoundaryT=new Real[8]{};
s.riemannBoundaryTFix=new int[8]{};
s.riemannBoundaryUFix=new int[8]{};
s.riemannBoundaryUx=new Real[8]{};
s.riemannBoundaryUy=new Real[8]{};
s.riemannBoundaryUz=new Real[8]{};
s.scheduledInletFaceMask=new int[8]{};
s.sstBoundaryK=new Real[8]{};
s.sstBoundaryKMode=new int[8]{};
s.sstBoundaryOmega=new Real[8]{};
s.sstBoundaryOmegaMode=new int[8]{};
s.sstF1=new Real[8]{};
s.sstF2=new Real[8]{};
s.sstPhiRhoK=new Real[8]{};
s.sstPhiRhoOmega=new Real[8]{};
s.sstWallDistance=new Real[8]{};
s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gasFluxScheme=1;s.gasReconstruction=0;s.gasLimiter=0;s.gasMu=0;s.gasPrClamped=.7;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.rhoMin=1e-12;s.TgasMin=1e-9;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
auto&g=s.gasSpecies;g.rho=new Real[4]{};g.initial=new Real[4]{};g.flux=new Real[6]{};g.gradX=new Real[4]{};g.gradY=new Real[4]{};g.gradZ=new Real[4]{};g.limiter=new Real[2]{};g.boundaryMassFraction=new Real[6]{};g.compositionBoundaryFixed=new int[3]{};g.positivityScale=new Real[4]{};g.soundSpeed=new Real[2]{};g.heatCapacity=new Real[2]{};g.gasConstant=new Real[2]{};g.cellStatus=new int[2]{};g.faceStatus=new int[3]{};
auto*meta=new ugkwp::SpeciesThermoData<Real>[2];auto*coeff=new Real[6]{1000,.1,-1e7,1100,.15,-2e7};g.thermo.species=meta;g.thermo.coefficients=coeff;g.thermo.coefficientCount=6;g.thermoControls.relativeEnergyTolerance=Real(2e-6);g.thermoControls.relativeTemperatureTolerance=Real(2e-6);
for(int k=0;k<2;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].referencePressure=101325;meta[k].coefficientOffset=3*k;}
g.rho[0]=.9;g.rho[1]=.1;g.rho[2]=.1;g.rho[3]=.9;
for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.rho[c]=1;Real rhos[2]={g.rho[c],g.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(rhos,Real(700),g.thermo);s.Tgas[c]=700;s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;}
s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;}
blockDim.x=1;blockIdx.x=0;
for(int c=0;c<2;++c){threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
}
void transport(State&s,double dt){
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,dt);}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);}
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);}
}

#include "gpu/GpuCouplingMath.H"
ThreadIndex gridDim;
struct PairedState:State {
 int count=3; int*particleCountDevice=&count; int preBaseCellOffset[3]={0,2,3};
 int particleCapacity=4,cellParticleOffset[3]={0,2,3},sortedParticleIndex[3]={0,1,2};
 int pStatus[4]={1,1,1,0},pCellId[4]={0,0,1,0};
 double pm[4]={.2,.4,.3,.1},pd[4]={.001,.0002,.0005,.001};
 double pux[4]={-10,30,2,999},puy[4]={3,-4,1,999},puz[4]={1,2,-2,999};
 double puxOld[4]={},puyOld[4]={},puzOld[4]={},pTheta[4]={2,4,3,999},pT[4]={300,1200,900,999};
 double epsGPrev[2]={.8,.6},rhoSolid=2500,particleThermalRho=2500,particleCp=800,particleThermalCapacity=2e6;
 double dragParameter0=.8,dragParameter1=0,dragParameter2=0,dragParameter3=0;
 double gasPrOneThird=std::cbrt(.71),TpMin=100,TpMax=2000;
 double gravityX=.2,gravityY=-.3,gravityZ=.4;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
};
#include "applications/gasUGKPSteady/private_backend/SteadyMixtureParticleCoupling.cuh"
void setup(PairedState&s){initialise(s);gridDim.x=1;s.gasMu=1.8e-5;s.gasPrClamped=.71;s.gasThermalConductivity=.06;s.V[0]=.7;s.V[1]=1.3;
for(int c=0;c<2;++c){s.rhoUx[c]=s.rho[c]*5;s.rhoUy[c]=s.rho[c]*-2;s.rhoUz[c]=s.rho[c]*1;s.rhoE[c]+=.5*s.rho[c]*(25+4+1);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}}
void closeValue(long double a,long double b,long double tol,const char*why){if(fabsl(a-b)>tol){std::fprintf(stderr,"%s: %.21Lg != %.21Lg difference %.21Lg\n",why,a,b,a-b);std::abort();}}
struct Budget {long double mass=0,x=0,y=0,z=0,energy=0;};
Budget budget(const PairedState&s){Budget b;
for(int c=0;c<2;++c){long double v=(long double)s.epsGPrev[c]*s.V[c];b.mass+=v*s.rho[c];b.x+=v*s.rhoUx[c];b.y+=v*s.rhoUy[c];b.z+=v*s.rhoUz[c];b.energy+=v*s.rhoE[c];}
for(int i=0;i<s.particleCapacity;++i)if(s.pStatus[i]==1){long double m=s.pm[i];b.mass+=m;b.x+=m*s.pux[i];b.y+=m*s.puy[i];b.z+=m*s.puz[i];b.energy+=m*(.5L*((long double)s.pux[i]*s.pux[i]+(long double)s.puy[i]*s.puy[i]+(long double)s.puz[i]*s.puz[i])+1.5L*s.pTheta[i]+800.L*s.pT[i]);}return b;}
void conserved(const Budget&a,const Budget&b){closeValue(a.mass,b.mass,1e-14L,"mass");closeValue(a.x,b.x,1e-12L,"momentum x");closeValue(a.y,b.y,1e-12L,"momentum y");closeValue(a.z,b.z,1e-12L,"momentum z");closeValue(a.energy,b.energy,5e-8L,"formation-inclusive total energy");}

#include "DiagnosticsHost.H"
#include "applications/gasUGKPSteady/private_backend/SteadySamplingMoments.cuh"
struct SampleState:PairedState {
  double epsS[2]{},rhoUsx[2]{},rhoUsy[2]{},rhoUsz[2]{},rhoEs[2]{},rhoHp[2]{},rhoDs[2]{};
  double steadySampleSum[32]{},steadySampleSquare[32]{},steadySampleMean[32]{};
  double steadySampleBegin[32]{},steadySampleScratch[160]{};
};
int main() {
  using namespace ugkp::diagnostics;
  SampleState s; setup(s);
  auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
  coeff[0]=coeff[3]=1000; coeff[1]=coeff[2]=coeff[4]=coeff[5]=0;
  s.count=1; s.pStatus[1]=s.pStatus[2]=0;
  s.cellParticleOffset[1]=s.cellParticleOffset[2]=1;
  s.preBaseCellOffset[1]=s.preBaseCellOffset[2]=1;
  s.pm[0]=1; s.pd[0]=.000001; s.pT[0]=10000;
  s.TpMin=100; s.TpMax=10000;
  s.pux[0]=s.puy[0]=s.puz[0]=s.pTheta[0]=0;
  s.gravityX=s.gravityY=s.gravityZ=0;
  s.gasSpecies.rho[0]=s.gasSpecies.rho[1]=.9;
  s.gasSpecies.rho[2]=s.gasSpecies.rho[3]=.1;
  s.epsS[0]=s.pm[0]/(s.rhoSolid*s.V[0]);s.epsS[1]=0;
  for(int c=0;c<2;++c)s.epsGPrev[c]=1-s.epsS[c];
  s.rhoHp[0]=s.pm[0]*800*s.pT[0]/s.V[0];s.rhoHp[1]=0;
  for(int c=0;c<2;++c) {
    s.rhoUx[c]=s.rhoUy[c]=s.rhoUz[c]=s.Ux[c]=s.Uy[c]=s.Uz[c]=0;
    Real partial[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
    s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(1000),s.gasSpecies.thermo);
    s.Tgas[c]=1000; threadIdx.x=c; recoverGasPrimitivesKernel(&s);
  }
  DeviceContext diagnostics{};
  s.diagnostics=&diagnostics;
  s.gasSpecies.thermo.diagnostics=&diagnostics;
  const double dt=0.00002, oldGas=s.rhoE[0],oldParcel=s.pT[0];
  const double volume=s.epsGPrev[0]*s.V[0];
  const double alpha=steadyMixtureParticles::heatFraction(s,0,0,dt);
  const double gasCapacity=volume*s.rhoE[0]/s.Tgas[0];
  const double parcelCapacity=s.pm[0]*800*alpha;
  const double expected=(gasCapacity*1000+parcelCapacity*10000)/(gasCapacity+parcelCapacity);
  for(int c=0;c<s.nCells;++c) {
    threadIdx.x=c;beginSteadySampleWindowKernel(&s);
  }
  threadIdx.x=0;
  steadyMixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,dt,false);
  const int statusAfterHeat=s.gasSpecies.cellStatus[0];
  const unsigned fatalAfterHeat=diagnostics.fatalClaim;
  const unsigned warningsAfterHeat=diagnostics.warnings[0].count+diagnostics.warnings[1].count+diagnostics.warnings[2].count;
  recoverGasPrimitivesKernel(&s);
  const int statusAfterRecovery=s.gasSpecies.cellStatus[0];
  const unsigned fatalAfterRecovery=diagnostics.fatalClaim;
  for(int f=0;f<s.nFaces;++f) {
    threadIdx.x=f;updateRiemannBoundaryMirrorKernel(&s);
  }
  const unsigned fatalAfterBoundary=diagnostics.fatalClaim;
  bool allSampleValuesFinite=true;
  for(int c=0;c<s.nCells;++c) {
    threadIdx.x=c;accumulateSteadySampleWindowKernel(&s,dt);
    finishSteadySampleWindowKernel(&s,dt);
  }
  for(double value:s.steadySampleScratch)allSampleValuesFinite=allSampleValuesFinite&&std::isfinite(value);
  const auto finalSnapshot=takeSnapshot(diagnostics,{},"review-host");
  const double sampleEnergyChange=s.steadySampleScratch[3*32+4*s.nCells]-s.steadySampleScratch[2*32+4*s.nCells];
  std::printf("{\"samplingReportFinite\":%s,\"samplingEnergyInventoryChange\":%.17g,\"completionSnapshotHasFatal\":%s,\"completionSnapshotWarnings\":%zu,",allSampleValuesFinite?"true":"false",sampleEnergyChange,finalSnapshot.hasFatal?"true":"false",finalSnapshot.warnings.size());
  threadIdx.x=0;
  computeGasInternalFaceFluxKernel<false>(&s,dt);
  std::printf("\"analyticRoot\":%.17g,\"gasMaximumTemperature\":3000,\"alpha\":%.17g,\"gasCapacity\":%.17g,\"parcelCapacity\":%.17g,\"statusAfterHeat\":%d,\"fatalAfterHeat\":%u,\"warningsAfterHeat\":%u,\"statusAfterRecovery\":%d,\"fatalAfterRecovery\":%u,\"fatalAfterBoundary\":%u,\"fatalAfterNextFlux\":%u,\"gasTemperature\":%.17g,\"parcelTemperature\":%.17g,\"gasEnergyChange\":%.17g,\"parcelTemperatureChange\":%.17g}\n",expected,alpha,gasCapacity,parcelCapacity,statusAfterHeat,fatalAfterHeat,warningsAfterHeat,statusAfterRecovery,fatalAfterRecovery,fatalAfterBoundary,diagnostics.fatalClaim,s.Tgas[0],s.pT[0],s.rhoE[0]-oldGas,s.pT[0]-oldParcel);
  ck(statusAfterHeat==int(ugkwp::ThermoStatus::TemperatureOutOfRange),"expected domain failure");
  ck(fatalAfterHeat==1 && warningsAfterHeat==0 && fatalAfterRecovery==1 && fatalAfterBoundary==1,"missing current-step diagnostic");
  ck(diagnostics.fatalClaim==1,"next flux did not consume failed cell");
  ck(s.rhoE[0]==oldGas && s.pT[0]==oldParcel,"failed heat was partially published");
}
