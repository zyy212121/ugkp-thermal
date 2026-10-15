#include "OpenFoamWallFunctions.cuh"
#include <cuda_runtime.h>
#include <iostream>
__global__ void checkPole(ugkpwall::JayatillekeThermalTransport* out) {
 const GpuReal rho=1.09552, cp=1722,mu=9.29152e-5,pr=0.4,prt=0.85,cmu=0.09,kap=0.41,E=9.8,k=33386.4,y=0.000149661,U=1309.67;
 const GpuReal P=ugkpwall::jayatillekeSmoothP(pr/prt), yt=ugkpwall::jayatillekeThermalYPlus(pr/prt,kap,E);
 const GpuReal u=sqrt(sqrt(cmu))*sqrt(k),yp=u*y*rho/mu,tp=prt*(log(E*yp)/kap+P),uc=u/kap*log(E*yt);
 const GpuReal C=GpuReal(0.5)*rho*u*(prt*U*U+(pr-prt)*uc*uc),mol=mu*cp/pr;
 GpuReal grad=-C/(mol*tp)*(GpuReal(1)+GpuReal(int(threadIdx.x)-1)*GpuReal(1e-6));
 out[threadIdx.x]=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,pr,prt,cmu,kap,E,P,yt,k,y,U,0,grad);
}
int main() {
 ugkpwall::JayatillekeThermalTransport *d,h[3];
 auto allocation=cudaMalloc(&d,sizeof(h)); if(allocation!=cudaSuccess){std::cerr<<cudaGetErrorString(allocation)<<"\n";return 1;}
 checkPole<<<1,3>>>(d);
 if(cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
 for(auto a:h) if(a.valid!=1||!std::isfinite(a.conductivity)||fabs(a.conductivity-2.54098)>0.001 || !std::isfinite(a.heatFlux))return 3;
 cudaFree(d);std::cout<<"PASS CUDA captured near-singular wall input: "<<h[1].conductivity<<"\n";
}
