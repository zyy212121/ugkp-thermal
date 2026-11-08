// Production sampling/CSR kernels: RNG, contact metadata, weighted moments.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) double(x)
#define GPU_OPERATOR_THERMAL 1
#define GPU_BLOCK_REDUCTION_PRUNED_TREE 0
#define GPU_CONTACT_AGE(s,i) ((s).pContactAge[i])
#define GPU_CONTACT_TIME_ZERO 0.
#define GPU_RESET_CONTACT_AGE(s,i) (s).pContactAge[i]=0.;
#define UGKP_SAMPLE_REDUCTION_DEVICE_ONLY
namespace Foam {namespace gpuThermal {constexpr int particleWallTransientRebound=2,particleWallTransientDeposit=3;}}
constexpr double OfGreat=1e30;
template<class T,class U> __device__ T clampMin(T x,U y){return x<y?T(y):x;}
template<class T,class U,class V> __device__ T clampRange(T x,U y,V z){return x<y?T(y):(x>z?T(z):x);}
template<class T,class U> __device__ T finiteOr(T x,U y){return isfinite(x)?x:T(y);}
__host__ __device__ double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
#include "sample_rng.inc"
constexpr int N=8193,C=5;
enum class CsrReductionTaskSource {fullIndexed=0,splitBaseDirect=1,splitInjectionIndexed=2,splitLogical=3};
struct CsrReductionTask {int cell,begin,end,source;};
struct DeviceState {
 int nCells=C,particleCapacity=N,active=N,selected=0,tasks=0,multi=0,cursor=0;
 int *particleCountDevice,*compactCountDevice,*csrHeavyTaskCount,*csrHeavyCellCount,*csrHeavyTaskCursor;
 int pStatus[N]{},pCellId[N]{},pStuck[N]{},pStuckFaceId[N]{},compactPStatus[N]{};
 unsigned long long pRng[N]{};
 double pux[N]{},puy[N]{},puz[N]{},pTheta[N]{},pm[N]{},pd[N]{},pContactAge[N]{};
 double pDepositionArea[N]{},pContactDuration[N]{},pContactMaximumArea[N]{},pContactPeakFraction[N]{};
 int poissonPoolSampleTargetCount[C]{},cellParticleOffset[C+1]{},preBaseCellOffset[C+1]{},sortedParticleIndex[N]{};
 int csrCellTaskCount[C+1]{},csrCellTaskOffset[C+1]{},csrMultiTaskCellList[C]{};
 CsrReductionTask csrReductionTasks[N]{};
 double csrHeavyPartials[8*N]{},thetaDragAlpha[C]{};
 double poissonPoolMass[C]{},poissonPoolMomX[C]{},poissonPoolMomY[C]{},poissonPoolMomZ[C]{},poissonPoolEnergy[C]{};
 double poolThermalSumUx[C]{},poolThermalSumUy[C]{},poolThermalSumUz[C]{},poolThermalSumU2[C]{};
 double particleDiameterMin=1e-6,particleDiameterMax=1e-3,particleDiameterFallback=1e-4;
};
#include "GpuBlockComponentReduction.cuh"
#include "operators/appendSelectedStuckParticleIndex.cuh"
#include "GpuCollisionPoolSampling.cuh"
#include "operators/samplePoissonPoolParticlesKernel.cuh"
#include "GpuPoissonSampleReduction.cuh"
void check(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(x));std::exit(2);}}
void require(bool x,const char* message){if(!x){fprintf(stderr,"%s\n",message);std::exit(1);}}
void bind(DeviceState* s) {
 s->particleCountDevice=&s->active;s->compactCountDevice=&s->selected;
 s->csrHeavyTaskCount=&s->tasks;s->csrHeavyCellCount=&s->multi;s->csrHeavyTaskCursor=&s->cursor;
}
std::unique_ptr<DeviceState> seed(int n,bool split,int selectedStride,int tile){
 auto s=std::make_unique<DeviceState>();s->active=n;
 std::vector<int> base[C],injected[C];
 const int baseCount=split?n*2/3:n;
 // Physical base is sorted by cell, injection directory is independently indexed.
 for(int i=0;i<n;++i) {
  const int c=i<baseCount?std::min(C-1,i*C/std::max(1,baseCount)):(i-baseCount)%C;
  s->pCellId[i]=c;s->pStatus[i]=i%selectedStride==0?2:1;
  s->pStuck[i]=i%4==0?2:(i%4==1?3:0);
  s->pStuckFaceId[i]=11;s->pContactAge[i]=.125;
  s->pContactDuration[i]=.5;s->pContactMaximumArea[i]=1e-8;s->pContactPeakFraction[i]=.4;
  s->pm[i]=1e-9*(1+i%5);s->pd[i]=1e-4;s->pRng[i]=123456789ULL+333*i;
  s->pux[i]=11;s->puy[i]=-7;s->puz[i]=4;s->pTheta[i]=.2;
  if(i<baseCount)base[c].push_back(i);else injected[c].push_back(i);
  if(s->pStatus[i]==2){s->poissonPoolMass[c]+=s->pm[i];++s->poissonPoolSampleTargetCount[c];}
 }
 int offset=0,boffset=0;
 for(int c=0;c<C;++c){
  s->cellParticleOffset[c]=offset;s->preBaseCellOffset[c]=boffset;
  if(split){for(int i:injected[c])s->sortedParticleIndex[offset++]=i;}
  else {for(int i:base[c])s->sortedParticleIndex[offset++]=i;}
  boffset+=base[c].size();
  s->csrCellTaskOffset[c]=s->tasks;
  const int first=split?0:s->cellParticleOffset[c],last=split?base[c].size()+injected[c].size():offset;
  for(int begin=first;begin<last;begin+=tile){
   s->csrReductionTasks[s->tasks++]={c,begin,std::min(last,begin+tile),split?3:0};
   ++s->csrCellTaskCount[c];
  }
  if(s->csrCellTaskCount[c]>1)s->csrMultiTaskCellList[s->multi++]=c;
  const double mass=s->poissonPoolMass[c];s->poissonPoolMomX[c]=11*mass;s->poissonPoolMomY[c]=-7*mass;
  s->poissonPoolMomZ[c]=4*mass;s->poissonPoolEnergy[c]=mass*(.5*sqr3(11,-7,4)+3);
  s->thetaDragAlpha[c]=.7;
 }
 s->cellParticleOffset[C]=offset;s->preBaseCellOffset[C]=boffset;s->csrCellTaskOffset[C]=s->tasks;return s;
}
std::unique_ptr<DeviceState> run(const DeviceState& input,int mode,bool split,int block,int drag){
 DeviceState* s;check(cudaMallocManaged(&s,sizeof(DeviceState)));*s=input;bind(s);
 const size_t shared=4*(block/32)*sizeof(double);
 if(mode==0)samplePoissonPoolParticlesKernel<<<24,block>>>(s,drag);
 if(mode==2){
  samplePoissonPoolTasksKernel<<<24,block,shared>>>(s,drag);
  check(cudaGetLastError());check(cudaMemsetAsync(&s->cursor,0,sizeof(int)));
  finalizePoissonSampleMomentsKernel<<<C,block,shared>>>(s);
 }
 check(cudaGetLastError());check(cudaDeviceSynchronize());
 auto out=std::make_unique<DeviceState>(*s);check(cudaFree(s));return out;
}
int main(){
 int tests=0;
 for(int n:{0,1,31,257,8193})for(bool split:{false,true})for(int stride:{1,13})for(int block:{32,128})for(int drag:{0,1})for(int emptyPool:{0,1}){
  auto initial=seed(n,split,stride,97);
  if(emptyPool){initial->poissonPoolSampleTargetCount[0]=0;initial->poissonPoolMass[1]=0;}
  auto reference=run(*initial,0,split,block,drag);
  for(int mode:{2}) {
   auto candidate=run(*initial,mode,split,block,drag);
   require(candidate->selected==reference->selected,"contact count");
   std::sort(candidate->compactPStatus,candidate->compactPStatus+candidate->selected);
   std::sort(reference->compactPStatus,reference->compactPStatus+reference->selected);
   for(int i=0;i<candidate->selected;++i)require(candidate->compactPStatus[i]==reference->compactPStatus[i],"contact indices");
   for(int i=0;i<n;++i) {
    require(candidate->pRng[i]==reference->pRng[i],"RNG");
    require(candidate->pStatus[i]==reference->pStatus[i],"status");
    require(candidate->pux[i]==reference->pux[i]&&candidate->puy[i]==reference->puy[i]&&candidate->puz[i]==reference->puz[i],"sample velocity");
    require(candidate->pStuck[i]==reference->pStuck[i]&&candidate->pStuckFaceId[i]==reference->pStuckFaceId[i],"contact identity");
    require(candidate->pContactAge[i]==reference->pContactAge[i]&&candidate->pContactDuration[i]==reference->pContactDuration[i]
       &&candidate->pContactMaximumArea[i]==reference->pContactMaximumArea[i]
       &&candidate->pContactPeakFraction[i]==reference->pContactPeakFraction[i],"contact metadata");
   }
   for(int c=0;c<C;++c) {
    double expected[4]={},scale[4]={};
    for(int i=0;i<n;++i)if(candidate->pStatus[i]==2&&candidate->pCellId[i]==c) {
     const double inv=1./candidate->poissonPoolMass[c];
     const double x=candidate->pux[i]-candidate->poissonPoolMomX[c]*inv,y=candidate->puy[i]-candidate->poissonPoolMomY[c]*inv,z=candidate->puz[i]-candidate->poissonPoolMomZ[c]*inv,m=candidate->pm[i];
     const double term[]={m*x,m*y,m*z,m*sqr3(x,y,z)};
     for(int k=0;k<4;++k){expected[k]+=term[k];scale[k]+=std::abs(term[k]);}
    }
    const double actual[]={candidate->poolThermalSumUx[c],candidate->poolThermalSumUy[c],candidate->poolThermalSumUz[c],candidate->poolThermalSumU2[c]};
    for(int k=0;k<4;++k)if(!(std::abs(actual[k]-expected[k])<2e-13*(1e-30+scale[k]))) {
     printf("n=%d split=%d stride=%d block=%d drag=%d mode=%d c=%d k=%d got=%.17g expected=%.17g\\n",n,split,stride,block,drag,mode,c,k,actual[k],expected[k]);require(false,"weighted moment");
    }
   }
   ++tests;
  }
 }
 printf("PASS %d sampling comparisons: RNG/velocity/contact exact; weighted moments checked independently\n",tests);
}
