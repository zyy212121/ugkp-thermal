// Execute shared production pool selection, accumulation, sampling, correction
// and wall finalization. Host adapters and CUDA kernels use the same fixture.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
struct Index {int x;}; Index threadIdx{0},blockIdx{0},blockDim{1},gridDim{1};
template<class T> T atomicAdd(T*p,T v){T old=*p;*p+=v;return old;}
#define asm(...) std::abort()
#endif
#include "wall/GpuColdWallSolidification.H"
#include "wall/GpuColdWall2DSolidification.H"
#include "wall/GpuFiniteWallContact.H"
#include "wall/GpuCapillaryDetachment.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) double(x)
#define GPU_OPERATOR_TINY(x) double(x)
#define GPU_OPERATOR_THERMAL 1
#define GPU_POOL_INITIALIZATION_WITH_PROBABILITY 0
#define GPU_POOL_MASS_FALLBACK 0.
#define GPU_POOL_PARTICLE_THETA(s,i) ((s).pStuck[i] ? 0. : (s).pTheta[i])
#define GPU_CONTACT_AGE(s,i) ((s).pContactAge[i])
#define GPU_CONTACT_TIME_ZERO 0.
#define GPU_CONTACT_SEPARATE_AGE 1
#define GPU_RESET_CONTACT_AGE(s,i) (s).pContactAge[i]=0.;
constexpr double OfSmall=1e-15,OfGreat=1e30,OfPi=3.14159265358979323846;
template<class T,class U> __device__ T clampMin(T x,U y){return x<y?T(y):x;}
template<class T,class U,class V> __device__ T clampRange(T x,U y,V z){return x<y?T(y):(x>z?T(z):x);}
template<class T,class U> __device__ T finiteOr(T x,U y){return std::isfinite(x)?x:T(y);}
__device__ bool finiteDevice(double x){return std::isfinite(x);}
__host__ __device__ double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
struct DeviceState {
 int nCells=1,nFaces=1,nInternalFaces=0,particleCapacity=2,count=2,selected=0;
 int *particleCountDevice=nullptr,*compactCountDevice=nullptr,compactPStatus[2]{};
 int pStatus[2]{1,1},pStuck[2]{2,0},pStuckFaceId[2]{0,-1},pCellId[2]{};
 unsigned char particleStuckCandidateMask[1]{};
 unsigned long long pRng[2]{1,2};
 double pm[2]{1e-8,1e-8},pd[2]{1e-4,1e-4},pT[2]{2000,2000},pTheta[2]{};
 double pux[2]{0,-20},puy[2]{},puz[2]{},puxOld[2]{3,0},puyOld[2]{4,0},puzOld[2]{};
  double pContactAge[2]{.4,0};
 double pContactDuration[2]{1,0},pContactPeakFraction[2]{.4,0};
 double pContactMaximumArea[2]{1e-8,0},pDepositionArea[2]{};
 double pColdFrozenArea[2]{1e-9,0},pCold2DFrozenArea[2]{1e-9,0},pColdContactAge[2]{.4,0};
 double pColdRingSolidMass[16]{},pCold2DRingContactAge[16]{};
 float pColdNodeSpecificEnthalpy[16]{},pCold2DNodeSpecificEnthalpy[128]{};
 double pThermalEnthalpyResidual[2]{.125,0};
 int coldWallSolidificationEnabled=1,coldWall2DEnabled=1;
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{};
 double particleWallAdhesionEnergyScale=1,particleWallContactAngleCosine=.5;
 double particleWallContactAreaScale[1]{1},magSf[1]{1},Sfx[1]{1},Sfy[1]{},Sfz[1]{};
 double gasBoundaryUx[1]{},gasBoundaryUy[1]{},gasBoundaryUz[1]{};
 double particleDiameterMin=1e-4,particleDiameterMax=1e-4,particleDiameterFallback=1e-4,particleDiameterN=1;
 double rhoSolid=2500,epsSMin=1e-15,thetaMin=1e-15;
 double momRhoP[1]{100},momRhoUPx[1]{-1000},momRhoUPy[1]{},momRhoUPz[1]{},momRhoEP[1]{10000},momRhoPD[1]{.01};
 int poolThermalCount[1]{},poissonPoolSampleTargetCount[1]{};
 double poissonPoolMass[1]{},poissonPoolMomX[1]{},poissonPoolMomY[1]{},poissonPoolMomZ[1]{},poissonPoolEnergy[1]{};
 double poissonPoolDiameter[1]{},poissonPoolDiameter2[1]{},poolThermalSumUx[1]{},poolThermalSumUy[1]{},poolThermalSumUz[1]{},poolThermalSumU2[1]{},thetaDragAlpha[1]{1};
};
#include "thermal/ParticleSurfaceState.H"
#include "GpuFrozenParticleRelease.cuh"
#include "operators/clearColdWallParticleState.cuh"
// Extracted verbatim from current owner by run_frozen_pool_release.py:
// RNG mixer, 2D profile clear, mobile publication. No physics replacements.
#include "owner_functions.inc"
#include "operators/uniform01Device.cuh"
#include "operators/normalDevice.cuh"
#include "operators/granularCollisionTauFromCellDevice.cuh"
#include "GpuCollisionPoolAtomic.cuh"
#include "operators/clearPoissonThermalPoolKernel.cuh"
#include "operators/preparePoissonPoolSamplingKernel.cuh"
#include "operators/appendSelectedStuckParticleIndex.cuh"
#include "GpuCollisionPoolSampling.cuh"
#include "operators/samplePoissonPoolParticlesKernel.cuh"
#include "GpuThermalParticleFinalization.cuh"
#include "operators/finalizeOneThermalizedParticlePath.cuh"
#include "GpuCollisionPoolCorrection.cuh"
__global__ void setup(DeviceState*s){if(threadIdx.x==0){s->particleCountDevice=&s->count;s->compactCountDevice=&s->selected;}}
__global__ void correct(DeviceState*s,bool witness){
 if(threadIdx.x!=0)return;
 // The deposit twin bypasses ONLY the new frozen-rebound zeroing branch.
 // Both finite states have exactly the same adhesion/reflection calculation.
 if(witness)s->pStuck[0]=Foam::gpuThermal::particleWallTransientDeposit;
 for(int i=0;i<2;++i)correctOnePoissonThermalizedParticlePath<false>(*s,i,false);
 for(int i=0;i<2;++i)correctOnePoissonThermalizedParticlePath<true>(*s,i,false);
}
#ifdef __CUDACC__
void check(cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(2);}}
#define RUN(kernel,...) do{kernel<<<1,32>>>(__VA_ARGS__);check(cudaGetLastError());check(cudaDeviceSynchronize());}while(0)
#else
#define RUN(kernel,...) kernel(__VA_ARGS__)
#endif
DeviceState advance(DeviceState input,bool witness){
#ifdef __CUDACC__
 DeviceState*s=nullptr;check(cudaMalloc(&s,sizeof(input)));check(cudaMemcpy(s,&input,sizeof(input),cudaMemcpyHostToDevice));
#else
 DeviceState*s=&input;
#endif
 RUN(setup,s);RUN(clearPoissonThermalPoolKernel,s);
 RUN(accumulateParticlePoolAtomicKernel<true>,s,.1);
 RUN(preparePoissonPoolSamplingKernel,s);RUN(samplePoissonPoolParticlesKernel,s,0);
 RUN(correct,s,witness);
#ifdef __CUDACC__
 check(cudaMemcpy(&input,s,sizeof(input),cudaMemcpyDeviceToHost));check(cudaFree(s));
#endif
 return input;
}
int main(){int failures=0;
 for(int model:{1,2})for(double sign:{-1.,1.})for(double saved:{0.,1.}) {
  DeviceState s;s.pContactAge[0]=.4;s.particleStuckCandidateMask[0]=model==1?Foam::gpuThermal::particleWallSolidifyingDeposition:Foam::gpuThermal::particleWallColdWall2D;
  s.pux[1]*=sign;s.momRhoUPx[0]*=sign;s.puxOld[0]*=saved;s.puyOld[0]*=saved;
  for(int n=0;n<8;++n)s.pColdNodeSpecificEnthalpy[n]=2546000+n;
  for(int n=0;n<64;++n)s.pCold2DNodeSpecificEnthalpy[n]=2546000+n;
  const int nodes=model==1?8:64;
  const double carry=.125;
  const double base=2546000+.5*(nodes-1)+carry+12.5*saved;
  const auto witness=advance(s,true),result=advance(s,false);
  assert(result.poolThermalCount[0]==2&&result.poissonPoolSampleTargetCount[0]==2&&result.selected==1&&result.compactPStatus[0]==0);
  assert(std::abs(result.poissonPoolEnergy[0]-2e-6)<1e-20);
  assert(witness.pStuck[0]==0&&result.pStuck[0]==0&&result.pStuckFaceId[0]==-1);
  assert(result.pContactAge[0]==0&&result.pContactDuration[0]==0);
  assert(sqr3(result.pux[0],result.puy[0],result.puz[0])==0);
  const double cancelled=.5*sqr3(witness.pux[0],witness.puy[0],witness.puz[0]);
  double represented=ugkwp::ParticleCaloric{}.enthalpy(result.pT[0]);
  represented+=result.pThermalEnthalpyResidual[0];
  const double error=(represented-base-cancelled)*s.pm[0];
  assert(cancelled>1.&&result.pux[1]==witness.pux[1]&&result.puy[1]==witness.puy[1]&&result.puz[1]==witness.puz[1]);
  std::printf("model=%d sign=%.0f saved=%.0f pool_J=%.17g cancelled_J=%.17g thermal_increment_J=%.17g error_J=%.17g\n",model,sign,saved,result.poissonPoolEnergy[0],cancelled*s.pm[0],(represented-base)*s.pm[0],error);
  if(std::abs(error)>1e-16)++failures;
 }
 return failures?1:0;
}
