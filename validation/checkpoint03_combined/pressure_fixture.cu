
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <cstddef>
#include "GpuPrecisionTypes.H"
using AcceptanceEnergy=GpuReal;
using PressureReal=AcceptanceEnergy;using PressureTime=double;using R=GpuReal;
#define GPU_OPERATOR_REAL R
#define GPU_OPERATOR_R(x) R(x)
#define GPU_OPERATOR_TIME double
#define GPU_BLOCK_REDUCTION_PRUNED_TREE 0
#include "GpuPressureProjectionTypes.H"
template<class A,class B> __device__ typename std::common_type<A,B>::type clampMin(A x,B a){return x<a?a:x;}
template<class T> __device__ T clampRange(T x,T a,T b){return x<a?a:x>b?b:x;}
template<class T> __device__ bool finiteDevice(T x){return (x==x)&&(fabs(x)<(sizeof(T)==8?1.0e300:1.0e30));}
template<class A,class B> __device__ typename std::common_type<A,B>::type finiteOr(A x,B a){return finiteDevice(x)?x:a;}
template<class T> __device__ T sqr3(T x,T y,T z){return x*x+y*y+z*z;}
constexpr R OfVSmall=1e-30;
struct DeviceState {
    int nCells=1,nFaces=1,particleCapacity=1100;
 int csrCellLocalPathEnabled=1; PressureProjectionCell pressureProjectionCache[2]={};
    int particleCount=2;
    int* particleCountDevice=&particleCount;
    R rhoMin=R(1e-20),particleDiameterFallback=1,TpMin=1,TpMax=1000;
    R epsSMin=R(1e-8),rhoSolid=1,thetaMin=R(.01),pressureKickFraction=10;
    R V[2]={1}; double cellLength[2]={1}; // Volume arithmetic follows Real; tracking length remains double.
    R momRhoP[2]={2},momRhoUPx[2]={0},momRhoUPy[2]={0},momRhoUPz[2]={0}; AcceptanceEnergy momRhoEP[2]={R(3.25)};
    R rhoUsx[2]={0},rhoUsy[2]={0},rhoUsz[2]={0}; AcceptanceEnergy rhoEs[2]={R(3.25)};
    R Usx[2]={0},Usy[2]={0},Usz[2]={0},theta[2]={R(3.25/3)},epsS[2]={2};
    int cellParticleCount[3]={2};
    R momRhoPD[2]={2}; AcceptanceEnergy momRhoHpP[2]={2};
    AcceptanceEnergy pressureParticleMoments[14]={}; int pressureParticleCount[2]={};
    R pd[1100]={1,1},pT[1100]={1,1},compactPd[1100]={1,1},compactPT[1100]={1,1};
    R puxOld[1100]={1,1},puyOld[1100]={1,1},puzOld[1100]={1,1};
    R compactPuxOld[1100]={1,1},compactPuyOld[1100]={1,1},compactPuzOld[1100]={1,1};
    int cellPlaneStart[2]={0},cellPlaneCount[2]={1},cellFaceId[2]={0};
    int faceOwner[2]={0},faceNeighbour[2]={-1};
    R solidPressurePhiMomX[2]={-1},solidPressurePhiMomY[2]={R(-.25)},solidPressurePhiMomZ[2]={R(.5)}; AcceptanceEnergy solidPressurePhiEnergy[2]={R(-.2)};
    R pressureDeltaMomX[2]={0},pressureDeltaMomY[2]={0},pressureDeltaMomZ[2]={0}; AcceptanceEnergy pressureDeltaEnergy[2]={0};
    R pressureKickScale[2]={0};
    int pStatus[1100]={1,1,0,0},pCellId[1100]={0,0,-1,-1},pStuck[1100]={0};
    R pm[1100]={1,1,0,0},pux[1100]={R(-.5),R(.5),0,0},puy[1100]={0},puz[1100]={0},pTheta[1100]={1,1,0,0};
    int compactPStatus[1100]={1,1,0,0},compactPCellId[1100]={0,0,-1,-1},compactPStuck[1100]={0};
    R compactPm[1100]={1,1,0,0},compactPux[1100]={R(-.5),R(.5),0,0},compactPuy[1100]={0},compactPuz[1100]={0},compactPTheta[1100]={1,1,0,0};
    int preBaseCellOffset[3]={0,0,0},cellParticleOffset[3]={0,2,4},sortedParticleIndex[1100]={0,1,2,3};
};
namespace Foam {namespace gpuThermal {constexpr int particleWallDeposited=1;}}
template<bool Compact> __device__ R particleMomentThetaDevice(const DeviceState&s,int i){
 if((Compact?s.compactPStuck[i]:s.pStuck[i])!=0)return 0;
 return clampMin(finiteOr(Compact?s.compactPTheta[i]:s.pTheta[i],R(0)),R(0));
}
__device__ R particleSpecificEnthalpyDevice(R t){return t;}
#include "GpuBlockComponentReduction.cuh"
#include "GpuPressureConstrainedPhysics.cuh"

#ifdef CP3_COUNT_PRESSURE
#define makePressureParameters cp3OriginalMakePressureParameters
#endif
#include "GpuPressureCellTraversal.cuh"
#ifdef CP3_COUNT_PRESSURE
#undef makePressureParameters
__device__ unsigned int cp3ParameterCalls;
template<class ParameterStorage>
__device__ __forceinline__ void makePressureParameters
(const DeviceState& s,const int c,const PressureReal (&delta)[4],ParameterStorage& q)
{
 atomicAdd(&cp3ParameterCalls,1u);
 cp3OriginalMakePressureParameters(s,c,delta,q);
}
#endif

#include "GpuPressureConstrainedAdapter.cuh"
#include "GpuPressureUnsortedConstrained.cuh"
#include "operators/applyCollisionalPressureProjectionKernel.cuh"
#include "GpuPressureAnalyticPhysics.cuh"

#define CUDA(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(err));std::exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);std::exit(1);}}while(0)

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>
void project(DeviceState*s,int route,int kind,int threads){
 size_t shared=7*((threads+31)/32)*sizeof(PressureReal);
 if(kind==0){
 if(route==1)applyAnalyticPressureProjectionKernel<true><<<1,threads>>>(s,1e-5);
 else if(route==2)applyAnalyticPressureProjectionKernel<false,true><<<1,threads>>>(s,1e-5);
 else applyAnalyticPressureProjectionKernel<false><<<1,threads>>>(s,1e-5);
 }else{
 if(route==1)applyUnifiedSplitPressureKernel<true><<<1,threads,shared>>>(s,1e-5);
 else if(route==2)applyCollisionalPressureProjectionKernel<true,true><<<1,threads,shared>>>(s,1e-5);
 else applyCollisionalPressureProjectionKernel<true><<<1,threads,shared>>>(s,1e-5);
 }
}
void initialise(DeviceState*s,int n,int route){
 new(s)DeviceState; s->particleCount=n;s->particleCountDevice=&s->particleCount;
 s->momRhoP[0]=n?R(2):R(0);s->momRhoUPx[0]=0;s->momRhoEP[0]=3;
 s->cellParticleOffset[1]=n;s->cellParticleCount[0]=n;
 if(route==1){s->preBaseCellOffset[1]=n/2;s->cellParticleOffset[0]=n/2;}
 for(int i=0;i<n;++i){
 s->sortedParticleIndex[i]=i;s->pStatus[i]=s->compactPStatus[i]=1;
 s->pCellId[i]=s->compactPCellId[i]=0;s->pm[i]=s->compactPm[i]=R(2)/R(n);
 s->pux[i]=s->compactPux[i]=0;s->puy[i]=s->compactPuy[i]=0;s->puz[i]=s->compactPuz[i]=0;
 s->pTheta[i]=s->compactPTheta[i]=1;s->pd[i]=s->compactPd[i]=1;s->pT[i]=s->compactPT[i]=1;
 }
}
int main(int argc,char**argv){
 bool perf=argc>1&&std::string(argv[1])=="perf";
 DeviceState*s;
 if(perf){CUDA(cudaMalloc(&s,sizeof(*s)));}
 else {CUDA(cudaMallocManaged(&s,sizeof(*s)));}
 auto reset=[&](int n,int route){
  if(perf){
   DeviceState staging;initialise(&staging,n,route);
   staging.particleCountDevice=reinterpret_cast<int*>(
     reinterpret_cast<char*>(s)+offsetof(DeviceState,particleCount));
   CUDA(cudaMemcpy(s,&staging,sizeof(staging),cudaMemcpyHostToDevice));
   CUDA(cudaDeviceSynchronize());
  }else initialise(s,n,route);
 };
 std::cout<<std::setprecision(17);
 bool first=true;int checks=0; double samples[5]={};
 std::ostringstream counts;bool firstCount=true;
 if(perf){
  // Same declared warmup in A and B, never selected from observed timings.
  const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
  unsigned long long warmupLaunches=0;
  do{reset(1025,0);for(int k=0;k<128;++k)project(s,0,0,128);CUDA(cudaDeviceSynchronize());warmupLaunches+=128;}
  while(std::chrono::steady_clock::now()<until);
  std::cout<<"WARMUP seconds_minimum=3 launches="<<warmupLaunches<<" population=1025 block=128 route=0\n";
 }
 if(!perf)std::cout<<"{\"family\":\"pressure\",\"precision\":"<<8*sizeof(R)<<",\"mode\":\"correctness\",\"measurements\":{";
 for(int block:{32,64,128})for(int n:{0,1,31,1025})for(int route=0;route<3;++route)for(int kind=0;kind<2;++kind){
 if(perf&&(kind!=0||n==0))continue;

 reset(n,route);
#ifdef CP3_COUNT_PRESSURE
 unsigned int zeroCalls=0;
 CUDA(cudaMemcpyToSymbol(cp3ParameterCalls,&zeroCalls,sizeof(zeroCalls)));
#endif
 project(s,route,kind,block);CUDA(cudaDeviceSynchronize());
#ifdef CP3_COUNT_PRESSURE
 unsigned int calls=0;CUDA(cudaMemcpyFromSymbol(&calls,cp3ParameterCalls,sizeof(calls)));
 unsigned int expectedCalls=kind==0?block:1;
#ifdef CP3_EXPECT_SHARED
 expectedCalls=1;
#endif
 CHECK(calls==expectedCalls);
 if(!firstCount)counts<<",";firstCount=false;
 counts<<"\""<<block<<"_"<<n<<"_"<<route<<"_"<<kind<<"\":"<<calls;
#endif

 if(perf){
 for(int warmup=0;warmup<32;++warmup)project(s,route,kind,block);CUDA(cudaDeviceSynchronize());
 cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));
 for(int j=0;j<5;++j){reset(n,route);CUDA(cudaEventRecord(a));for(int r=0;r<100;++r)project(s,route,kind,block);CUDA(cudaEventRecord(b));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));samples[j]+=ms/1000.;}
 CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));
 }else{
 std::string key=std::to_string(block)+"_"+std::to_string(n)+"_"+std::to_string(route)+"_"+std::to_string(kind);
 auto emit=[&](std::string name,double x){CHECK(std::isfinite(x));if(!first)std::cout<<",";first=false;std::cout<<"\""<<key<<"_"<<name<<"\":"<<x;++checks;};
 emit("momentum",s->momRhoUPx[0]);emit("energy",s->momRhoEP[0]);emit("theta",s->theta[0]);
 double mx=0,en=0;
 for(int i=0;i<n;++i){R u=route==2?s->compactPux[i]:s->pux[i],v=route==2?s->compactPuy[i]:s->puy[i],w=route==2?s->compactPuz[i]:s->puz[i],t=route==2?s->compactPTheta[i]:s->pTheta[i];
 emit("ux"+std::to_string(i),u);emit("theta"+std::to_string(i),t);mx+=double(s->pm[i])*u;en+=double(s->pm[i])*(.5*(u*u+v*v+w*w)+1.5*t);}
 emit("parcel_momentum",mx);emit("parcel_energy",en);
 if(n){double tol=sizeof(R)==4?5e-4:1e-9;CHECK(std::abs(mx-s->momRhoUPx[0])<tol);CHECK(std::abs(en-s->momRhoEP[0])<tol*4);}
 }
 }
 if(perf){std::cout<<"{\"family\":\"pressure\",\"precision\":"<<8*sizeof(R)<<",\"mode\":\"perf\",\"samples_seconds\":[";for(int i=0;i<5;++i)std::cout<<(i?",":"")<<samples[i];std::cout<<"],\"iterations\":100,\"worksize\":27,\"populations\":[1,31,1025],\"blocks\":[32,64,128],\"routes\":3,\"warmup_per_case\":33,\"warmup_process_seconds_minimum\":3,\"actual_shared_bytes_per_block\":"<<sizeof(AnalyticPressureOperation::Shared)<<",\"parameter_struct_bytes\":"<<sizeof(PressureParameters)<<",\"global_cache_bytes\":0,\"setup/reset_excluded\":true,\"cache_bytes\":0}\n";}
 else std::cout<<"},\"counts\":{"<<counts.str()<<"},\"checks\":"<<checks<<"}\n";
 CUDA(cudaFree(s));
}
