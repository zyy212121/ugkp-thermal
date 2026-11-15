
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
using PressureReal=TEST_REAL;
using PressureTime=double;
using R=PressureReal;
#define GPU_OPERATOR_REAL PressureReal
#define GPU_OPERATOR_R(x) PressureReal(x)
#define GPU_BLOCK_REDUCTION_PRUNED_TREE 0
#include "GpuPressureProjectionTypes.H"
template<class T> __host__ __device__ T clampMin(T x,T y){return x>y?x:y;}
template<class T> __host__ __device__ T clampRange(T x,T lo,T hi){return x<lo?lo:x>hi?hi:x;}
template<class T> __host__ __device__ bool finiteDevice(T x){return isfinite(x);}
template<class T> __host__ __device__ T finiteOr(T x,T fallback){return finiteDevice(x)?x:fallback;}
template<class T> __host__ __device__ T sqr3(T x,T y,T z){return x*x+y*y+z*z;}
constexpr R OfVSmall=R(1e-30);
struct DeviceState {
    int nCells=1,nFaces=1,particleCapacity=4;
    int csrCellLocalPathEnabled=1;
    int pressureCellHasContact[2]={};
    PressureProjectionCell pressureProjectionCache[2]={};
    int particleCount=2;
    int* particleCountDevice=&particleCount;
    R rhoMin=R(1e-20),particleDiameterFallback=1,TpMin=1,TpMax=1000;
    R epsSMin=R(1e-8),rhoSolid=1,thetaMin=R(.01),pressureKickFraction=10;
    R V[2]={1},cellLength[2]={1};
    R momRhoP[2]={2},momRhoUPx[2]={0},momRhoUPy[2]={0},momRhoUPz[2]={0},momRhoEP[2]={R(3.25)};
    R rhoUsx[2]={0},rhoUsy[2]={0},rhoUsz[2]={0},rhoEs[2]={R(3.25)};
    R Usx[2]={0},Usy[2]={0},Usz[2]={0},theta[2]={R(3.25/3)},epsS[2]={2};
    int cellParticleCount[3]={2};
    R momRhoPD[2]={2},momRhoHpP[2]={2};
    R pressureParticleMoments[14]={}; int pressureParticleCount[2]={};
    R pd[4]={1,1},pT[4]={1,1},compactPd[4]={1,1},compactPT[4]={1,1};
    R puxOld[4]={1,1},puyOld[4]={1,1},puzOld[4]={1,1};
    R compactPuxOld[4]={1,1},compactPuyOld[4]={1,1},compactPuzOld[4]={1,1};
    int cellPlaneStart[2]={0},cellPlaneCount[2]={1},cellFaceId[2]={0};
    int faceOwner[2]={0},faceNeighbour[2]={-1};
    R solidPressurePhiMomX[2]={-1},solidPressurePhiMomY[2]={R(-.25)},solidPressurePhiMomZ[2]={R(.5)},solidPressurePhiEnergy[2]={R(-.2)};
    R pressureDeltaMomX[2]={0},pressureDeltaMomY[2]={0},pressureDeltaMomZ[2]={0},pressureDeltaEnergy[2]={0};
    R pressureKickScale[2]={0};
    int pStatus[4]={1,1,0,0},pCellId[4]={0,0,-1,-1},pStuck[4]={0};
    R pm[4]={1,1,0,0},pux[4]={R(-.5),R(.5),0,0},puy[4]={0},puz[4]={0},pTheta[4]={1,1,0,0};
    int compactPStatus[4]={1,1,0,0},compactPCellId[4]={0,0,-1,-1},compactPStuck[4]={0};
    R compactPm[4]={1,1,0,0},compactPux[4]={R(-.5),R(.5),0,0},compactPuy[4]={0},compactPuz[4]={0},compactPTheta[4]={1,1,0,0};
    int preBaseCellOffset[3]={0,0,0},cellParticleOffset[3]={0,2,4},sortedParticleIndex[4]={0,1,2,3};
};

namespace Foam {namespace gpuThermal {constexpr int particleWallDeposited=1;}}
template<bool Compact> __device__ R particleMomentThetaDevice(const DeviceState&s,int i){
    if((Compact?s.compactPStuck[i]:s.pStuck[i])!=0)return 0;
    return clampMin(finiteOr(Compact?s.compactPTheta[i]:s.pTheta[i],R(0)),R(0));
}
__device__ R particleSpecificEnthalpyDevice(R t){return t;}
#include "GpuBlockComponentReduction.cuh"
#include "GpuPressureLimiter.cuh"
#include "GpuPressureAnalyticPhysics.cuh"
#include "GpuPressureConstrainedPhysics.cuh"
#include "GpuPressureConstrainedAdapter.cuh"
#include "GpuPressureUnsortedConstrained.cuh"
#include "GpuPressureContactProjection.cuh"

template<bool Compact> __global__ void cellReference(DeviceState* s,double dt){
    const int c=blockIdx.x;
    if(pressureDirectoryContainsContact<PressureDirectory::Full,Compact>(*s,c))
        runCellPressureProjection<ConstrainedPressureOperation<false>,PressureDirectory::Full,Compact>(s,dt);
    else runCellPressureProjection<AnalyticPressureOperation,PressureDirectory::Full,Compact>(s,dt);
}
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
void near(R a,R b,const char* label,int i){
    const R tol=R(512)*std::numeric_limits<R>::epsilon()*(R(1)+fabs(b));
    if(!std::isfinite(double(a))||fabs(a-b)>tol){
        fprintf(stderr,"%s[%d] %.17g != %.17g\n",label,i,double(a),double(b));exit(3);
    }
}
template<bool Compact> void trial(bool contact,bool mixed,bool analytic=false){
    DeviceState initial;
    initial.particleCount=4;
    initial.pStatus[2]=initial.compactPStatus[2]=1;
    initial.pCellId[2]=initial.compactPCellId[2]=0;
    initial.pux[2]=initial.compactPux[2]=R(321);
    if(contact)initial.pStuck[0]=initial.compactPStuck[0]=1;
    if(mixed){
        initial.nCells=2;
        initial.cellParticleOffset[2]=4;
        initial.cellParticleCount[1]=2;
        initial.momRhoP[1]=2;initial.momRhoEP[1]=3;
        initial.V[1]=1;initial.cellLength[1]=1;
        for(int i=2;i<4;++i){
            initial.pStatus[i]=initial.compactPStatus[i]=1;
            initial.pCellId[i]=initial.compactPCellId[i]=1;
            initial.pm[i]=initial.compactPm[i]=1;
            initial.pux[i]=initial.compactPux[i]=R(i-2)-R(.5);
            initial.pTheta[i]=initial.compactPTheta[i]=1;
        }
    }else if(!Compact){
        initial.particleCount=2;
    }
    DeviceState *old,*now;
    check(cudaMallocManaged(&old,sizeof(DeviceState)));check(cudaMallocManaged(&now,sizeof(DeviceState)));
    *old=initial;*now=initial;
    old->particleCountDevice=&old->particleCount;now->particleCountDevice=&now->particleCount;
    cellReference<Compact><<<initial.nCells,128,4*4*sizeof(R)>>>(old,.01);
    if(analytic){
        preparePressureProjectionCacheKernel<<<1,128>>>(now,.01);
        applyCachedPressureProjectionParticlesKernel<Compact><<<1,128>>>(now);
    }else{
        applyContactAwarePressureProjectionKernel<false,false,Compact>
            <<<initial.nCells,128,4*4*sizeof(R)>>>(now,.01);
        applyFreePressureProjectionParticlesKernel<Compact><<<1,128>>>(now);
    }
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    for(int i=0;i<4;++i){
        near(Compact?now->compactPux[i]:now->pux[i],Compact?old->compactPux[i]:old->pux[i],"ux",i);
        near(Compact?now->compactPTheta[i]:now->pTheta[i],Compact?old->compactPTheta[i]:old->pTheta[i],"theta",i);
    }
    for(int c=0;c<initial.nCells;++c){
        near(now->momRhoUPx[c],old->momRhoUPx[c],"cellPx",c);
        near(now->momRhoEP[c],old->momRhoEP[c],"cellEnergy",c);
    }
    check(cudaFree(old));check(cudaFree(now));
}
int main(){
    trial<true>(false,false,true);trial<false>(false,false,true);
    trial<true>(false,false);trial<true>(true,false);trial<true>(true,true);
    trial<false>(false,false);trial<false>(true,true);
    puts("PASS production pressure CUDA compact/free/contact/mixed/tail");
}
