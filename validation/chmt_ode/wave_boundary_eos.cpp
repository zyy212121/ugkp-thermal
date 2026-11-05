// Regression derived from the independent CHMT-2EB-R01 production-function probe.
// Serial execution of actual shared kernels, checked against the independently
// rearranged implicit wave-pressure equation. Full CUDA coverage is the CHMT run.
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
#include "gasTransport/GasStateView.H"
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
struct TestState:ugkwp::GasStateView<Real,Time,decltype(ugkwp::defaultSstCoefficients())>{ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;};
#include <iostream>
#include <stdexcept>

int main(){
 TestState s{};s.nCells=1;s.nFaces=1;s.nInternalFaces=0;s.Rgas=287;s.rhoMin=1e-12;s.TgasMin=100;
 ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
#define REAL_FIELD(name,value) Real name[]={value};s.name=name
#define INT_FIELD(name,value) int name[]={value};s.name=name
 INT_FIELD(faceOwner,0);REAL_FIELD(magSf,1);REAL_FIELD(Sfx,1);REAL_FIELD(Sfy,0);REAL_FIELD(Sfz,0);
 REAL_FIELD(deltaCoeffs,20);REAL_FIELD(rho,2);REAL_FIELD(Ux,0);REAL_FIELD(Uy,0);REAL_FIELD(Uz,0);REAL_FIELD(p,300000);
 REAL_FIELD(gasBoundaryUx,0);REAL_FIELD(gasBoundaryUy,0);REAL_FIELD(gasBoundaryUz,0);INT_FIELD(gasBoundaryUFix,0);
 INT_FIELD(gasBoundaryKind,0);INT_FIELD(gasBoundaryRhoFix,1);INT_FIELD(gasBoundaryTFix,0);INT_FIELD(gasBoundaryPFix,0);REAL_FIELD(Tgas,600);
 REAL_FIELD(gasBoundaryRho,2);REAL_FIELD(gasBoundaryP,150000);REAL_FIELD(gasBoundaryT,600);
 INT_FIELD(gasBoundaryPWave,1);REAL_FIELD(gasBoundaryPWaveGamma,1.2);REAL_FIELD(gasBoundaryPWaveLInf,0);REAL_FIELD(gasBoundaryPWaveFieldInf,101325);
 REAL_FIELD(riemannBoundaryP,150000);REAL_FIELD(riemannBoundaryT,600);REAL_FIELD(riemannBoundaryRho,2);
 REAL_FIELD(riemannBoundaryUx,0);REAL_FIELD(riemannBoundaryUy,0);REAL_FIELD(riemannBoundaryUz,0);
 INT_FIELD(riemannBoundaryPWave,1);INT_FIELD(riemannBoundaryPFix,0);INT_FIELD(riemannBoundaryTFix,0);INT_FIELD(riemannBoundaryRhoFix,1);INT_FIELD(riemannBoundaryUFix,0);
 INT_FIELD(riemannBoundaryKind,0);
 ugkwp::SpeciesThermoData<Real> species[2]{};Real coeff[]={1200,0,0,1500,0,0};
 for(int k=0;k<2;++k){species[k].molarMass=k?.044:.028;species[k].coefficientOffset=3*k;species[k].minTemperature=100;species[k].maxTemperature=5000;}
 s.gasSpecies.thermo.species=species;s.gasSpecies.thermo.coefficients=coeff;s.gasSpecies.thermo.coefficientCount=6;
 Real density[]={1.5,.5},boundaryY[]={.2,.8};int fixed[]={0};s.gasSpecies.rho=density;s.gasSpecies.boundaryMassFraction=boundaryY;s.gasSpecies.compositionBoundaryFixed=fixed;
 s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;blockIdx.x=threadIdx.x=0;blockDim.x=1;
 ugkp::boundary::Definition def;def.faces.resize(1);def.faces[0].rhoMode=1;def.faces[0].tMode=0;def.faces[0].wave=1;def.faces[0].waveGamma=1.2;ugkp::boundary::validate(def);
 const double ownerY[2]={.75,.25};p[0]=2*ugkwp::mixtureGasConstant(ownerY,s.gasSpecies.thermo)*600;
 for(int fixedY=0;fixedY<2;++fixedY){
   fixed[0]=fixedY;gasBoundaryP[0]=riemannBoundaryP[0]=150000;
   const double ys[2]={fixedY?.2:.75,fixedY?.8:.25};
   const double R=ugkwp::mixtureGasConstant(ys,s.gasSpecies.thermo);
   gasBoundaryT[0]=riemannBoundaryT[0]=900; // stale value must be replaced by actual mirror operators
   updateLegacyGasBoundaryMirrorKernel(&s,0.);
   updateRiemannBoundaryMirrorKernel(&s,0.);
   if(gasBoundaryT[0]!=600 || riemannBoundaryT[0]!=600)return 4;
   const auto q=riemannBoundaryState(s,0,GasPrimDevice{2,0,0,0,p[0],600});
   const double expectedT=150000/(2*R),waveOld=gasBoundaryP[0];
   if(ugkp::diagnostics::hasFatal(&diagnostics)||std::abs(q.rho-2)>1e-12||std::abs(q.T-expectedT)>1e-10)return 1;
   const double actualSpeed=std::sqrt(1.2*R*gasBoundaryT[0]);
   const double boundarySpeed=std::sqrt(1.2*waveOld/2);
   const double expected=(waveOld+boundarySpeed*1e-5*20*p[0])/(1+boundarySpeed*1e-5*20);
   updateWaveTransmissivePressureBoundaryKernel(&s,1e-5);
   const double observed=gasBoundaryP[0];
   const double emulated=(waveOld+actualSpeed*1e-5*20*p[0])/(1+actualSpeed*1e-5*20);
   std::printf("Y_policy=%s R=%.9f rho_fixed=2 oldP=150000 ownerT=600 ownerP=%.9f EOS_T=%.9f stale_mirror_speed=%.9f speed_EOS=%.9f pressure_kernel=%.9f pressure_EOS_reference=%.9f\n",fixedY?"fixed":"owner",R,p[0],q.T,actualSpeed,boundarySpeed,observed,expected);
   if(std::abs(observed-expected)>1e-8)return 2;
 }
 return 0;
}
