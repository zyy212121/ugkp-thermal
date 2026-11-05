// Exercise the actual shared mirror kernels on CUDA. The adapter deliberately
// has no resident total-condition program: these are scheduled legacy inlets.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "gasTransport/GasStateView.H"
#include "gasTransport/MixtureThermo.H"
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) double(x)
constexpr double OfSmall=1e-15;
__device__ double clampMin(double a,double b){return fmax(a,b);}
__device__ double clampRange(double a,double b,double c){return fmin(fmax(a,b),c);}
__device__ double finiteOr(double a,double b){return isfinite(a)?a:b;}
struct State {
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    int nCells=1,nInternalFaces=0,nFaces=4,nScheduledInletFaces=2;
    int scheduledInletFaceMask[4]={1,1,0,1};
    int nPressureScheduleRows=2;
    double pressureScheduleTimes[2]={0,1},pressureScheduleValues[2]={4000000,4000000};
    double scheduledInletTemperature=3500,TgasMin=100,rhoMin=1e-12,Rgas=300;
    int faceOwner[4]={0,0,0,0};
    double rho[1]={2},Ux[1]={800},Uy[1]={-3},Uz[1]={4},p[1]={100000},Tgas[1]={300};
#define MIRROR(prefix) \
    int prefix##Kind[4]={0,0,0,4},prefix##RhoFix[4]={0,0,0,0},prefix##UFix[4]={0,1,0,0}; \
    int prefix##TFix[4]={0,0,0,0},prefix##PFix[4]={0,0,0,0},prefix##PWave[4]={0,0,0,0}; \
    double prefix##Rho[4]={1,1,1,1},prefix##Ux[4]={0,7,0,9},prefix##Uy[4]={0,8,0,10}; \
    double prefix##Uz[4]={0,9,0,11},prefix##P[4]={1,1,1,1},prefix##T[4]={1,1,1,1};
    MIRROR(gasBoundary)
    MIRROR(riemannBoundary)
#undef MIRROR
};
template<class S> __device__ int coupledFaceNeighbour(const S&,int){return 0;}
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/evaluateResidentBoundary.cuh"
#include "mirror_kernels.inc"
void checkCuda(cudaError_t status){if(status!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));std::exit(2);}}
int main(){
    State* s;checkCuda(cudaMallocManaged(&s,sizeof(State)));*s=State{};
    int errors=0,checks=0;
    auto same=[&](const char* name,double actual,double expected){
        ++checks;if(std::abs(actual-expected)>1e-12*std::fmax(1.,std::abs(expected))){
            std::printf("FAIL %s actual=%.17g expected=%.17g\n",name,actual,expected);++errors;}
    };
    for(int step=0;step<3;++step){
        s->pressureScheduleValues[1]=step==0?4000000:6000000;
        s->Ux[0]=step==0?800:step==1?1200:-50;
        s->Uy[0]=step+2;s->Uz[0]=-step-3;
        updateLegacyGasBoundaryMirrorKernel<<<1,32>>>(s,.5);
        checkCuda(cudaGetLastError());checkCuda(cudaDeviceSynchronize());
        same("scheduled owner Ux",s->gasBoundaryUx[0],s->Ux[0]);
        same("scheduled owner Uy",s->gasBoundaryUy[0],s->Uy[0]);
        same("scheduled owner Uz",s->gasBoundaryUz[0],s->Uz[0]);
        for(int f=0;f<2;++f){
            double pressure=step==0?4000000:5000000;
            same("scheduled pressure",s->gasBoundaryP[f],pressure);
            same("scheduled temperature",s->gasBoundaryT[f],3500);
            same("scheduled density",s->gasBoundaryRho[f],pressure/(300*3500.));
            same("Riemann scheduled pressure",s->riemannBoundaryP[f],pressure);
        }
        same("fixed Ux",s->gasBoundaryUx[1],7);
        same("fixed Uy",s->gasBoundaryUy[1],8);
        same("fixed Uz",s->gasBoundaryUz[1],9);
        same("ordinary owner Ux",s->gasBoundaryUx[2],s->Ux[0]);
        same("ordinary pressure",s->gasBoundaryP[2],100000);
        same("ordinary temperature",s->gasBoundaryT[2],300);
        same("ordinary density",s->gasBoundaryRho[2],2);
        same("empty velocity",s->gasBoundaryUx[3],9);
    }
    checkCuda(cudaFree(s));std::printf("checks=%d errors=%d\n",checks,errors);
    return errors?1:0;
}
