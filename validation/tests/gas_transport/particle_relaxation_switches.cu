// Actual shared relaxation on CPU or CUDA; inactive wall routes trap if entered.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#define __device__
#define __forceinline__ inline
#endif
#include "gasTransport/GasStateView.H"
#include "wall/GpuColdWallSolidification.H"
#include "wall/GpuFiniteWallContact.H"
#include "wall/GpuCapillaryDetachment.H"
#include "wall/GpuParticleWallContactHeat.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "gasNumerics/GpuDragModelsFSHCHT.cuh"
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) double(x)
#define GPU_OPERATOR_TINY(x) double(x)
#ifndef GPU_THERMAL_RELAX_NATIVE_ORDER
#define GPU_THERMAL_RELAX_NATIVE_ORDER 0
#endif
#define GPU_CONTACT_AGE(s,i) s.pContactAge[i]
#define GPU_CONTACT_TIME_ZERO double(0)
#define GPU_RESET_CONTACT_AGE(s,i) s.pContactAge[i]=0;
constexpr double OfSmall=1e-15;
__device__ double clampMin(double x,double lo){return x<lo?lo:x;}
__device__ double clampRange(double x,double lo,double hi){return x<lo?lo:(x>hi?hi:x);}
__device__ double finiteOr(double x,double fallback){return ugkwp::gasFinite(x)?x:fallback;}
__device__ double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
struct DeviceState:ugkwp::GasStateView<double,double,int> {
 int particleCapacity=1,solveParticleTemperature=1,particleGasHeatTransferModelId=0,dragModelId=0;
 int particleWallHeatTransferEnabled=0,pCellId[1]={0},pStuckFaceId[1]={0};
 unsigned char pStatus[1]={1},pStuck[1]={0},particleStuckCandidateMask[1]={0};
 double pd[1]={.0002},pm[1]={1e-8},pT[1]={250},pTheta[1]={.05};
 double couplingRhoOld[1]={1.2},couplingUxOld[1]={3},couplingUyOld[1]={-2},couplingUzOld[1]={1};
 double* couplingTgasOld=nullptr;
 double pux[1]={1},puy[1]={2},puz[1]={-1},puxOld[1]={},puyOld[1]={},puzOld[1]={};
 double gradPx[1]={},gradPy[1]={},gradPz[1]={},thetaDragAlpha[1]={.75};
 double particleDiameterFallback=.0002,TpMin=1,TpMax=5000,gasPrOneThird=.9;
 double rhoSolid=2000,invRhoSolid=.0005,gravityX=0,gravityY=0,gravityZ=0;
 double pContactAge[1]={},pContactDuration[1]={},pContactMaximumArea[1]={},pContactPeakFraction[1]={},pDepositionArea[1]={},pColdFrozenArea[1]={};
 double* particleWallEffusivityByFace=nullptr;
 double particleWallDensityKgM3=0,particleWallSpecificHeatJkgK=0,particleWallConductivityWmK=0;
 double particleWallReflectionHeatTransferEfficiency=0,particleWallContactAreaScale[1]={};
 double particleWallAdhesionEnergyScale=0,particleWallContactAngleCosine=0;
 double particleWallReflectedEnergy[1]={};
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{};
};
__device__ bool gasDragModelActive(int model){return model!=0;}
__device__ double solidEpsFromMomentDevice(const DeviceState&,int){return 0.;}
__device__ void unusedWallRoute(){
#if defined(__CUDA_ARCH__)
 asm("trap;");
#else
 std::abort();
#endif
}
__device__ void atomicAddParticleWallEnergyByFace(DeviceState&,double*,int,double){unusedWallRoute();}
__device__ void clearColdWallParticleState(DeviceState&,int){unusedWallRoute();}
__device__ void clearColdWall2DParticleState(DeviceState&,int){unusedWallRoute();}
#include "gasTransport/GasMolecularConductivity.cuh"
#include "operators/particleSpecificHeatDevice.cuh"
#include "operators/dragInverseTimeDevice.cuh"
#if !defined(__CUDACC__)
#define asm(...) std::abort()
#endif
#include "GpuThermalParticleRelaxation.cuh"

__device__ void evaluate(int index,double* row){
 const int mode=index%4,stuck=(index/4)%2,t=index/8;
 double temperature=300.+300.*t;
 ugkwp::SutherlandSpecies<double> species{1.8e-5,300.,110.4,.028};
 DeviceState s;s.nCells=1;s.nFaces=1;s.rhoMin=1e-12;s.TgasMin=1.;
 s.gasViscosity={ugkwp::MolecularViscosityModel::SutherlandWilke,1,&species,nullptr,nullptr};
 s.gasMu=1.8e-5;s.gasCp=1000.;s.gasPrClamped=.72;s.gasPrOneThird=::cbrt(.72);
 s.dragModelId=mode&1;s.particleGasHeatTransferModelId=(mode>>1)&1;
 // Disabled models must not dereference the absent temperature snapshot.
 s.couplingTgasOld=mode?&temperature:nullptr;
 s.pStuck[0]=stuck?Foam::gpuThermal::particleWallDeposited:Foam::gpuThermal::particleWallMobile;
 row[0]=mode;row[1]=stuck;row[2]=temperature;row[3]=particleSpecificHeatDevice(s.pT[0]);
 relaxOneParticleToResidentGas(s,0,1e-4,ugkwpGpuDrag::SchillerNaumannDrag{});
 row[4]=s.pux[0];row[5]=s.puy[0];row[6]=s.puz[0];row[7]=s.pT[0];row[8]=s.pTheta[0];
}
#if defined(__CUDACC__)
__global__ void run(double* result,int activeOnly){
 const int i=threadIdx.x;if(i>=24||(activeOnly&&i%4==0))return;
 evaluate(i,result+9*i);
}
#endif
int main(int argc,char**argv){
 const bool activeOnly=argc>1&&std::strcmp(argv[1],"--active-only")==0;
 double result[24*9]={};
#if defined(__CUDACC__)
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
 double* output=nullptr;if(cudaMalloc(&output,sizeof(result))!=cudaSuccess)return 1;
 run<<<1,32>>>(output,activeOnly);
 if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(result,output,sizeof(result),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
 cudaFree(output);
#else
 for(int i=0;i<24;++i)if(!activeOnly||i%4!=0)evaluate(i,result+9*i);
#endif
 for(int i=0;i<24;++i)if(!activeOnly||i%4!=0){
  for(int j=0;j<9;++j)std::printf("%.17g%c",result[i*9+j],j==8?'\n':' ');
 }
}
