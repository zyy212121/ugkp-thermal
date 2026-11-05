#include "ParticleDiameterDistribution.H"
#include "wall/GpuFiniteWallContact.H"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
__global__ void sample(double* d,int* release,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double u=(i+.5)/count;
 d[i]=ugkwp::rosinRammlerDiameter(1e-4,2.5,3e-5,4e-4,u);
 const auto r=Foam::gpuThermal::reboundContactRelease(i%2?0.:1.,i%3?.25:0.,0.);
 release[i]=(r.detach?1:0)+(r.atRest?2:0);
}
int main(){
 constexpr int n=65536;double* d;int* r;
 assert(cudaMallocManaged(&d,n*sizeof(double))==cudaSuccess);
 assert(cudaMallocManaged(&r,n*sizeof(int))==cudaSuccess);
 sample<<<256,256>>>(d,r,n);assert(cudaDeviceSynchronize()==cudaSuccess);
 double maximum=0;
 const long double a=std::pow(.3L,2.5L),b=std::pow(4.L,2.5L);
 for(int i=0;i<n;++i){
  const long double cdf=-std::expm1(-(std::pow((long double)d[i]/1e-4L,2.5L)-a))/(-std::expm1(-(b-a)));
  const double error=std::abs(double(cdf)-(i+.5)/n);maximum=std::fmax(maximum,error);
  assert(std::isfinite(d[i])&&d[i]>=3e-5&&d[i]<=4e-4&&error<2e-13);
  assert(r[i]==(i%2?1:0)+(i%3?2:0));
 }
 printf("PASS CUDA RR %d inverse-CDF samples max_error=%.17g; frozen/free release flags\n",n,maximum);
 assert(cudaFree(d)==cudaSuccess);assert(cudaFree(r)==cudaSuccess);
}
