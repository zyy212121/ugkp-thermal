"""Real CUDA pressure projection: compare shared gas targets and contact closure.
Usage: python3 pressure_native.py NEW_OUTPUT_DIRECTORY
"""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[2];O=Path(sys.argv[1]).resolve();O.mkdir(parents=True,exist_ok=False)
f=(R/'validation/tests/fixtures/pressure/projection_cpu.cpp').read_text()
a=f.index('struct DeviceState {');b=f.index('\n};',a)+3
state=f[a:b].replace('    int nCells=1,nFaces=1,particleCapacity=4;','''    int nCells=1,nFaces=1,particleCapacity=4;
    int csrCellLocalPathEnabled=1;
    int pressureCellHasContact[2]={};
    PressureProjectionCell pressureProjectionCache[2]={};''')
source=r'''
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
using PressureReal=double;using PressureTime=double;using R=double;
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
#define GPU_OPERATOR_TIME double
#define GPU_BLOCK_REDUCTION_PRUNED_TREE 0
#include "GpuPressureProjectionTypes.H"
template<class T> __device__ T clampMin(T x,T a){return x<a?a:x;}
template<class T> __device__ T clampRange(T x,T a,T b){return x<a?a:x>b?b:x;}
template<class T> __device__ bool finiteDevice(T x){return isfinite(x);}
template<class T> __device__ T finiteOr(T x,T a){return finiteDevice(x)?x:a;}
template<class T> __device__ T sqr3(T x,T y,T z){return x*x+y*y+z*z;}
constexpr double OfVSmall=1e-30;
'''+state+r'''
namespace Foam {namespace gpuThermal {constexpr int particleWallDeposited=1;}}
template<bool Compact> __device__ R particleMomentThetaDevice(const DeviceState&s,int i){
 if((Compact?s.compactPStuck[i]:s.pStuck[i])!=0)return 0;
 return clampMin(finiteOr(Compact?s.compactPTheta[i]:s.pTheta[i],0.),0.);
}
__device__ R particleSpecificEnthalpyDevice(R t){return t;}
#include "GpuBlockComponentReduction.cuh"
#include "GpuPressureConstrainedPhysics.cuh"
#include "GpuPressureCellTraversal.cuh"
#include "GpuPressureConstrainedAdapter.cuh"
#include "GpuPressureUnsortedConstrained.cuh"
#include "operators/applyCollisionalPressureProjectionKernel.cuh"
#include "GpuPressureAnalyticPhysics.cuh"
#include "GpuPressureContactProjection.cuh"
#define CUDA(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(err));std::exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);std::exit(1);}}while(0)
template<bool Full>
void project(DeviceState*s,int route,int kind,int threads){
 const size_t shared=(Full?7:4)*((threads+31)/32)*sizeof(double);
 if(route==3){
  s->csrCellLocalPathEnabled=0;
  if(kind==0){preparePressureProjectionCacheKernel<<<1,threads>>>(s,.1);applyCachedPressureProjectionParticlesKernel<<<1,threads>>>(s);}
  else if(kind==1){
   applyCollisionalPressureProjectionCellAtomicKernel<<<1,threads>>>(s,.1);
   applyCollisionalPressureProjectionParticlesAtomicKernel<Full><<<1,threads>>>(s);
   publishPressureParticleMomentsKernel<Full><<<1,threads>>>(s);
  }else{
   markPressureContactCellsKernel<<<1,threads>>>(s);
   preparePressureProjectionCacheKernel<<<1,threads>>>(s,.1);
   applyContactAwarePressureUnsortedKernel<Full><<<1,threads>>>(s);
   publishContactPressureMomentsKernel<Full><<<1,threads>>>(s);
  }
 }else if(kind==0){
  if(route==1)applyAnalyticPressureProjectionKernel<true><<<1,threads>>>(s,.1);
  else if(route==2)applyAnalyticPressureProjectionKernel<false,true><<<1,threads>>>(s,.1);
  else applyAnalyticPressureProjectionKernel<false><<<1,threads>>>(s,.1);
 }else if(kind==1){
  if(route==1)applyUnifiedSplitPressureKernel<Full><<<1,threads,shared>>>(s,.1);
  else if(route==2)applyCollisionalPressureProjectionKernel<Full,true><<<1,threads,shared>>>(s,.1);
  else applyCollisionalPressureProjectionKernel<Full><<<1,threads,shared>>>(s,.1);
 }else{
  if(route==1)applyContactAwarePressureProjectionKernel<Full,true><<<1,threads,shared>>>(s,.1);
  else if(route==2)applyContactAwarePressureProjectionKernel<Full,false,true><<<1,threads,shared>>>(s,.1);
  else applyContactAwarePressureProjectionKernel<Full,false><<<1,threads,shared>>>(s,.1);
 }
 CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
}
void near(double x,double y){if(std::abs(x-y)>2e-13*std::max(1.,std::abs(y))){fprintf(stderr,"%.17g != %.17g\n",x,y);std::exit(1);}}
template<bool Full>int cases(){int tested=0;
 for(int route=0;route<4;++route)for(int contact=0;contact<3;++contact)
 for(int empty=0;empty<2;++empty)for(int cold=0;cold<2;++cold)for(int selected=0;selected<2;++selected)for(int threads:{32,64,128}){
  DeviceState init;init.pStatus[0]=init.compactPStatus[0]=selected?2:1;init.momRhoUPx[0]=.05;init.momRhoEP[0]=cold?.001:3.25;
  if(empty)init.momRhoP[0]=0;
  init.pStuck[0]=contact>0;init.pStuck[1]=contact>1;
  for(int i=0;i<2;++i){init.pTheta[i]=cold?0:1;init.compactPStuck[i]=init.pStuck[i];init.compactPTheta[i]=init.pTheta[i];}
  if(route==1){init.preBaseCellOffset[1]=1;init.cellParticleOffset[0]=1;}
  DeviceState *reference,*candidate;
  CUDA(cudaMallocManaged(&reference,sizeof(DeviceState)));CUDA(cudaMallocManaged(&candidate,sizeof(DeviceState)));
  new(reference)DeviceState(init);new(candidate)DeviceState(init);
  reference->particleCountDevice=&reference->particleCount;candidate->particleCountDevice=&candidate->particleCount;
  project<Full>(reference,route,contact?1:0,threads);project<Full>(candidate,route,2,threads);
#define CELL(field) near(candidate->field[0],reference->field[0])
  CELL(momRhoP);CELL(momRhoPD);CELL(momRhoHpP);CELL(momRhoUPx);CELL(momRhoUPy);CELL(momRhoUPz);CELL(momRhoEP);
  CELL(rhoUsx);CELL(rhoUsy);CELL(rhoUsz);CELL(rhoEs);CELL(Usx);CELL(Usy);CELL(Usz);CELL(theta);
#undef CELL
  for(int i=0;i<2;++i){
#define PARCEL(field) near(route==2?candidate->compactP##field[i]:candidate->p##field[i],route==2?reference->compactP##field[i]:reference->p##field[i])
   PARCEL(ux);PARCEL(uy);PARCEL(uz);PARCEL(Theta);
#undef PARCEL
  }
  CUDA(cudaFree(reference));CUDA(cudaFree(candidate));++tested;
 }
 return tested;
}
int main(){printf("PASS native pressure: %d configurations, shared analytic free-cell targets and preserved contact closure\n",cases<true>()+cases<false>());}
'''
f=O/'pressure.cu';f.write_text(source);exe=O/'pressure'
with (O/'build.log').open('w') as log:
    p=subprocess.run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','-arch=sm_89','-I'+str(R/'common'),str(f),'-o',str(exe)],stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,(O/'build.log').read_text()[-6000:]
with (O/'memcheck.log').open('w') as log:
    p=subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','9',str(exe)],stdout=log,stderr=subprocess.STDOUT)
print((O/'memcheck.log').read_text());assert p.returncode==0
