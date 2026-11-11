#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "GpuPrecisionTypes.H"
#include "GpuCollisionCentralMoment.H"
#define GPU_OPERATOR_REAL float
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) float(x)
#define GPU_OPERATOR_TINY(x) 1.0e-30f
#ifndef GPU_OPERATOR_THERMAL
#define GPU_OPERATOR_THERMAL 0
#endif
#define GPU_POOL_PARTICLE_THETA(s,i) ((s).pTheta[i])
#define GPU_POOL_MASS_FALLBACK 0.0f
#define GPU_POOL_THETA_AFTER_REJECTION 0
#define GPU_POOL_STATIC_DIRECTORY 0
#define GPU_POOL_SPLIT_REDUCTION_TOPOLOGY PoolReductionTopology::warpComponents
#define GPU_POOL_CACHED_PROBABILITY 0
#define GPU_POOL_SPLIT_DIRECTORY (descriptor.source==3)
#define GPU_POOL_DIRECT_PARTICLE_INDEX (descriptor.source==1)
#define GPU_POOL_SAMPLING_READY(s,c)
#define GPU_POOL_END_TASK(s,c,first) ((s).tasks)
#define UGKP_SAMPLE_REDUCTION_DEVICE_ONLY
#define UGKP_POOL_REDUCTION_DEVICE_ONLY
#define GPU_CONTACT_AGE(s,i) ((s).pContactAge[i])
#define GPU_CONTACT_TIME_ZERO 0.0
#define GPU_RESET_CONTACT_AGE(s,i) (s).pContactAge[i]=0.0;
constexpr float OfSmall=2.220446e-16f,OfGreat=1.0f/OfSmall,OfPi=3.141592653589793f;
namespace Foam {namespace gpuThermal {constexpr int particleWallTransientRebound=2,particleWallTransientDeposit=3;}}
template<class T,class U> __device__ T clampMin(T a,U b){return a<T(b)?T(b):a;}
template<class T,class U,class V> __device__ T clampRange(T a,U b,V c){return a<T(b)?T(b):a>T(c)?T(c):a;}
template<class T,class U> __device__ T finiteOr(T a,U b){return isfinite(a)?a:T(b);}
__device__ bool finiteDevice(float a){return isfinite(a);}
__device__ float sqr3(float x,float y,float z){return x*x+y*y+z*z;}
constexpr int N=4097;
enum class CsrReductionTaskSource {fullIndexed=0,splitBaseDirect=1,splitInjectionIndexed=2,splitLogical=3};
struct CsrReductionTask {int cell,begin,end,source;};
struct DeviceState {
 int csrHeavyReductionEnabled=0;
 int nCells=1,particleCapacity=N,active=0,tasks=0,multi=0,cursor=0,contacts=0;
 int *particleCountDevice,*csrHeavyTaskCount,*csrHeavyCellCount,*csrHeavyTaskCursor;
 float pux[N]{},puy[N]{},puz[N]{},pTheta[N]{},pm[N]{},pd[N]{};
 int pStatus[N]{},pCellId[N]{},pStuck[N]{},pStuckFaceId[N]{};
 unsigned long long pRng[N]{};
 double pContactAge[N]{};
 float pDepositionArea[N]{},pContactDuration[N]{},pContactMaximumArea[N]{},pContactPeakFraction[N]{};
 int cellParticleOffset[2]{},preBaseCellOffset[2]{},sortedParticleIndex[N]{};
 int poissonPoolSampleTargetCount[1]{},poolThermalCount[1]{};
 int csrCellTaskCount[1]{},csrCellTaskOffset[2]{},csrMultiTaskCellList[1]{};
 CsrReductionTask csrReductionTasks[N]{};
 float csrCollisionPartials[8*N]{};
 CollisionCentralMoment collisionCellMoments[1]{};
 float poissonPoolMass[1]{},poissonPoolMomX[1]{},poissonPoolMomY[1]{},poissonPoolMomZ[1]{},poissonPoolEnergy[1]{};
 float poissonPoolDiameter[1]{},poissonPoolDiameter2[1]{};
 float poolThermalMass[1]{},poolThermalSumUx[1]{},poolThermalSumUy[1]{},poolThermalSumUz[1]{},poolThermalSumU2[1]{},thetaDragAlpha[1]{};
 float momRhoP[1]{1},momRhoPD[1]{1e-4f},rhoSolid=2500,epsSMin=1e-12f,thetaMin=1e-20f;
 float particleDiameterMin=1e-6f,particleDiameterMax=1e-3f,particleDiameterFallback=1e-4f,particleDiameterN=3;
 float tau=0;
};
__device__ unsigned long long mixSeed(unsigned long long x)
{
 x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
 x=(x^(x>>27))*0x94d049bb133111ebULL;
 return x^(x>>31);
}
#include "operators/uniform01Device.cuh"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "operators/normalDevice.cuh"
#include "operators/granularCollisionTauFromCellDevice.cuh"
#include "GpuCollisionPoolParticle.cuh"
#include "GpuPoolMomentOperations.cuh"
#include "GpuCollisionPoolRange.cuh"
#include "GpuCollisionPoolCell.cuh"
#include "GpuCollisionPoolSplitCell.cuh"
#include "GpuCollisionPoolSplitParticle.cuh"
#include "operators/accumulateCsrSplitLogicalPoolTask.cuh"
#include "GpuCollisionProbability.cuh"
#include "GpuSegmentedPoolWorkers.cuh"
#include "GpuCollisionPoolAtomic.cuh"
#include "GpuCollisionPoolTarget.cuh"
__device__ void appendSelectedStuckParticleIndex(DeviceState&,int){}
#include "GpuCollisionPoolSampling.cuh"
#include "GpuPoissonSampleReduction.cuh"
#if GPU_OPERATOR_THERMAL
template<bool Stuck> __device__ void finalizeOneThermalizedParticlePath(
 DeviceState& s,int i,float x,float y,float z,float,float,float)
{
 // This fixture verifies routing, not the separately tested wall contact law.
 if(Stuck)atomicAdd(&s.contacts,1);
 s.pux[i]=x;s.puy[i]=y;s.puz[i]=z;s.pTheta[i]=0;s.pStatus[i]=1;
}
#endif
#include "GpuCollisionPoolCorrection.cuh"
__global__ void prepare(DeviceState* s){if(!threadIdx.x){preparePoissonPoolSamplingCell(*s,0);s->tau=granularCollisionTauFromCellDevice(*s,0);}}
__global__ void correct(DeviceState* s,int drag){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
#if GPU_OPERATOR_THERMAL
 if(i<s->active&&s->pStuck[i])correctOnePoissonThermalizedParticlePath<true>(*s,i,drag);
 else correctOnePoissonThermalizedParticlePath<false>(*s,i,drag);
#else
 correctOnePoissonThermalizedParticle(*s,i,drag);
#endif
}
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
void require(bool b,const char* m){if(!b){std::fprintf(stderr,"%s\n",m);std::exit(1);}}
void init(DeviceState* s,int n,int directory,float boost)
{
 *s=DeviceState{};s->active=n;s->particleCountDevice=&s->active;
 s->csrHeavyTaskCount=&s->tasks;s->csrHeavyCellCount=&s->multi;s->csrHeavyTaskCursor=&s->cursor;
 const int base=directory==1?n/2:directory==2?n:0;
 s->preBaseCellOffset[1]=base;s->cellParticleOffset[1]=n-base;
 for(int i=0;i<n;++i){
  s->pStatus[i]=1;s->pm[i]=1e-9f*(1+i%7);s->pux[i]=boost+float(i%7)*0.0625f;
  s->puy[i]=float(i%5)*0.125f;s->puz[i]=-float(i%3)*0.0625f;
  s->pTheta[i]=0.01f;s->pd[i]=1e-4f;s->pRng[i]=123456789ULL+333*i;
 }
 for(int i=base;i<n;++i)s->sortedParticleIndex[i-base]=i;
 for(int start=0;start<n;start+=97)s->csrReductionTasks[s->tasks++]={0,start,std::min(n,start+97),directory?3:0};
 s->csrCellTaskCount[0]=s->tasks;s->csrCellTaskOffset[1]=s->tasks;s->multi=s->tasks>1;
 s->thetaDragAlpha[0]=1.0f;
}
void pool(DeviceState* s,int directory,int block,bool heavy,bool atomic){
 const size_t sh=8*((block+31)/32)*sizeof(float);
 reduceCollisionCellCentralKernel<false><<<1,block,sh>>>(s,directory);
 if(atomic){
  accumulateParticlePoolAtomicKernel<true><<<17,block>>>(s,1e8);
  reduceCollisionCellCentralKernel<true><<<1,block,sh>>>(s,directory);
 }else if(heavy){
  check(cudaMemsetAsync(&s->cursor,0,sizeof(int)));
  accumulateCsrSegmentedPoolTasksPersistentKernel<true><<<4,block,sh>>>(s,1e8);
  check(cudaMemsetAsync(&s->cursor,0,sizeof(int)));
  finalizeCsrSegmentedPoolCellsKernel<<<1,block,sh>>>(s);
 }else {
  if(!directory)accumulatePoissonPoolParticlesByCellKernel<<<1,block,sh>>>(s,1e8);
  else {
   accumulatePoissonPoolSplitSegmentKernel<true,false><<<1,block,sh>>>(s,1e8);
   if(directory==1)accumulatePoissonPoolSplitSegmentKernel<false,true><<<1,block,sh>>>(s,1e8);
  }
 }
 prepare<<<1,1>>>(s);check(cudaGetLastError());check(cudaDeviceSynchronize());
}
void sample(DeviceState* s,int directory,int block,bool heavy,int drag){
 // Cache immutable launch geometry before GPU work makes managed state inaccessible to the host.
 const int correctGrid=(s->active+block-1)/block;
 const size_t sh=5*((block+31)/32)*sizeof(float);
 if(heavy){
  check(cudaMemsetAsync(&s->cursor,0,sizeof(int)));
  samplePoissonPoolTasksKernel<<<4,block,sh>>>(s,drag);
  check(cudaMemsetAsync(&s->cursor,0,sizeof(int)));
  finalizePoissonSampleMomentsKernel<<<1,block,sh>>>(s);
 }else samplePoissonPoolCellsCentralKernel<<<1,block,sh>>>(s,directory,drag);
 correct<<<correctGrid,block>>>(s,drag);
 check(cudaGetLastError());check(cudaDeviceSynchronize());
}
int main(){
 DeviceState* s;check(cudaMallocManaged(&s,sizeof(DeviceState)));
 int count=0;double maxMassRel=0,maxMeanAbs=0,maxEnergyRel=0;
 init(s,0,0,0);
 reduceCollisionCellCentralKernel<false><<<1,32,8*sizeof(float)>>>(s,0);
 reduceCollisionCellCentralKernel<true><<<1,32,8*sizeof(float)>>>(s,0);
 samplePoissonPoolCellsCentralKernel<<<1,32,5*sizeof(float)>>>(s,0,0);
 check(cudaGetLastError());check(cudaDeviceSynchronize());
 require(s->collisionCellMoments[0].mass==0&&s->poissonPoolMass[0]==0&&s->poolThermalMass[0]==0,"empty state");

 for(int n:{1,3,31,1025,4097})for(int directory:{0,1,2})for(int block:{32,128})
 for(int mode:{0,1,2})for(float boost:{0.0f,1000.0f}){
  const bool heavy=mode==1;
  init(s,n,directory,boost);pool(s,directory,block,heavy,mode==2);
  const float mass=s->poissonPoolMass[0],q=s->poissonPoolEnergy[0];
  const float mx=s->poissonPoolMomX[0],my=s->poissonPoolMomY[0],mz=s->poissonPoolMomZ[0];
  require(s->tau>0&&s->tau<1e6f,"high-speed cold collision time vanished");
  require(q>0,"central Q vanished");require(s->poissonPoolSampleTargetCount[0]==n,"pool count");
  sample(s,directory,block,heavy,0);
  double m=0,x=0,y=0,z=0,e=0;
  for(int i=0;i<n;++i){m+=s->pm[i];x+=double(s->pm[i])*s->pux[i];y+=double(s->pm[i])*s->puy[i];z+=double(s->pm[i])*s->puz[i];}
  x/=m;y/=m;z/=m;
  for(int i=0;i<n;++i)e+=double(s->pm[i])*(.5*((s->pux[i]-x)*(s->pux[i]-x)+(s->puy[i]-y)*(s->puy[i]-y)+(s->puz[i]-z)*(s->puz[i]-z))+1.5*s->pTheta[i]);
  maxMassRel=std::max(maxMassRel,std::abs(m-mass)/mass);
  maxMeanAbs=std::max(maxMeanAbs,std::max(std::abs(x-mx),std::max(std::abs(y-my),std::abs(z-mz))));
  maxEnergyRel=std::max(maxEnergyRel,std::abs(e-q)/q);
  require(std::abs(m-mass)<mass*2e-5,"mass conservation");
  require(std::abs(x-mx)<2e-4&&std::abs(y-my)<2e-5&&std::abs(z-mz)<2e-5,"mean conservation");
  require(std::abs(e-q)<q*0.004,"central energy conservation");
  ++count;
 }
 // Repeat the real sample/correct singleton path without rebuilding theta.
 init(s,1,0,1000);const float theta=s->pTheta[0],ux=s->pux[0];
 for(int i=0;i<10000;++i){
  pool(s,0,32,false,true);const auto rng=s->pRng[0];sample(s,0,32,false,0);
  require(s->pTheta[0]==theta&&s->pux[0]==ux,"singleton identity drift");
  require(s->pRng[0]!=rng,"singleton failed to advance RNG");
 }
 pool(s,0,32,false,true);s->thetaDragAlpha[0]=0.7f;sample(s,0,32,false,1);
 require(s->pTheta[0]==theta*0.7f,"physical theta drag lost");
#if GPU_OPERATOR_THERMAL
 init(s,1,0,1000);s->pStuck[0]=2;pool(s,0,32,false,true);sample(s,0,32,false,0);
 require(s->contacts==1,"contact singleton bypassed physics");
#endif
 check(cudaFree(s));std::printf("PASS FP32 central moments %d cases; singleton identity and drag; max mass relative=%.9g mean absolute=%.9g Q relative=%.9g\n",count,maxMassRel,maxMeanAbs,maxEnergyRel);
}
