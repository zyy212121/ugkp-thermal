// Exercise the actual common segmented moment kernel with mixed empty/full cells.
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
struct DeviceState {
 int nCells,particleCapacity,csrHeavyReductionEnabled;
 int *cellParticleOffset,*cellParticleCount,*sortedParticleIndex,*pStatus,*pCellId,*csrHeavyTaskCursor;
 double *V,*pm,*pux,*puy,*puz,*pTheta,*pd,*pT;
 double *momRhoP,*momRhoUPx,*momRhoUPy,*momRhoUPz,*momRhoEP,*momRhoPD,*momRhoHpP;
 double rhoMin=.125,TpMin=200,TpMax=4000,particleDiameterFallback=.001;
};
template<class T> __device__ T clampMin(T a,T b){return a<b?b:a;}
template<class T> __device__ T clampRange(T a,T b,T c){return clampMin(a,b)>c?c:clampMin(a,b);}
template<class T> __device__ T finiteOr(T a,T b){return isfinite(a)?a:b;}
__device__ double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
__device__ double particleHeatFactorDevice(const DeviceState&){return 2.;}
__device__ double particleSpecificEnthalpyDevice(double t){return 2*t;}
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R double
#define GPU_BLOCK_REDUCTION_PRUNED_TREE 1
#include "common/GpuBlockComponentReduction.cuh"
#define GPU_MOMENT_REAL double
#define GPU_MOMENT_R double
#define GPU_MOMENT_THERMAL 0
#define GPU_MOMENT_GATHER 0
#include "common/GpuParticleMoments.cuh"
#include "common/CsrPersistentQueue.cuh"
std::vector<void*> allocations;
template<class T> void alloc(T*& p,int n){assert(cudaMallocManaged(&p,n*sizeof(T))==cudaSuccess);allocations.push_back(p);}
__global__ void checkReset(int* counter,int* result){if(threadIdx.x==0)*result=*counter;}
int main(){
 DeviceState *s;alloc(s,1);*s=DeviceState{};
 const int counts[]={0,1,31,0,32,33,129,0};s->nCells=8;s->particleCapacity=226;
 alloc(s->cellParticleOffset,9);alloc(s->cellParticleCount,10);
 alloc(s->sortedParticleIndex,226);alloc(s->pStatus,226);alloc(s->pCellId,226);
 for(auto p:{&s->V,&s->momRhoP,&s->momRhoUPx,&s->momRhoUPy,&s->momRhoUPz,&s->momRhoEP,&s->momRhoPD,&s->momRhoHpP})alloc(*p,9);
 for(auto p:{&s->pm,&s->pux,&s->puy,&s->puz,&s->pTheta,&s->pd,&s->pT})alloc(*p,226);
 int pos=0;s->cellParticleOffset[0]=0;
 for(int c=0;c<8;++c){s->V[c]=c==0?.01:2.;for(int j=0;j<counts[c];++j,++pos){
  s->sortedParticleIndex[pos]=pos;s->pStatus[pos]=1;s->pCellId[pos]=c;
  s->pm[pos]=.25;s->pux[pos]=2;s->puy[pos]=3;s->puz[pos]=4;s->pTheta[pos]=.5;s->pd[pos]=.125;s->pT[pos]=300;
 }s->cellParticleOffset[c+1]=pos;}assert(pos==226);
 for(int block:{32,64,96,128,256}){
  double* fields[]={s->momRhoP,s->momRhoUPx,s->momRhoUPy,s->momRhoUPz,s->momRhoEP,s->momRhoPD,s->momRhoHpP};
  for(auto p:fields)std::fill(p,p+9,-777.);std::fill(s->cellParticleCount,s->cellParticleCount+10,-777);
  accumulateParticleMomentsSegmentedKernel<false><<<8,block,8*((block+31)/32)*sizeof(double)>>>(s);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  const double perParticle[]={.25,.5,.75,1.,.25*(.5*29+1.5*.5),.25*.125,150.};
  for(int c=0;c<8;++c){assert(s->cellParticleCount[c]==counts[c]);for(int f=0;f<7;++f)assert(fields[f][c]==counts[c]*perParticle[f]/std::max(s->V[c],s->rhoMin));}
  assert(s->cellParticleCount[8]==0&&s->cellParticleCount[9]==-777);for(auto p:fields)assert(p[8]==-777.);
 }
 alloc(s->csrHeavyTaskCursor,1);int *result;alloc(result,1);
 const DeviceState host=*s;
 for(int j=0;j<100;++j){assert(cudaMemsetAsync(host.csrHeavyTaskCursor,0x7f,sizeof(int))==cudaSuccess);assert(resetCsrPersistentQueue(&host)==cudaSuccess);checkReset<<<1,1>>>(host.csrHeavyTaskCursor,result);assert(cudaDeviceSynchronize()==cudaSuccess);assert(*result==0);}
 for(auto p:allocations)cudaFree(p);
 puts("EMPTY_MOMENT_PASS mixed_cells=8 block_sizes=5 ordered_resets=100");
}
