#include "common/GpuGasCourantReduction.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>
#include <chrono>
struct OldMaximum {
 __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
void check(cudaError_t e){assert(e==cudaSuccess);}
int main(){
 for(int streamMode=0;streamMode<2;++streamMode){
 cudaStream_t stream=nullptr;if(streamMode)check(cudaStreamCreate(&stream));
 for(int n: {0,1,31,32,127,128,255,256,1025,65537})for(int mode=0;mode<6;++mode){
  std::vector<double> h(3*(n+1));
  for(int j=0;j<3;++j)for(int i=0;i<n;++i)h[j*(n+1)+i]=(i*13+j*71)%101-50.;
  if(mode==1)for(auto& x:h)x=-0.;
  if(mode==5)for(int i=0;i<int(h.size());++i)h[i]=i%2?0.:-0.;
  if(mode==2&&n)for(int j=0;j<3;++j)h[j*(n+1)+n/2]=INFINITY;
  if(mode==3&&n){for(int j=0;j<3;++j){h[j*(n+1)+n/3]=nan("11");h[j*(n+1)+n/2]=nan("29");}}
  double* d=nullptr;check(cudaMalloc(&d,h.size()*sizeof(double)));check(cudaMemcpy(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice));
  double expected[3]={0,0,0};
  for(int j=0;j<(mode==4?2:3);++j)expected[j]=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(d+j*(n+1)),thrust::device_pointer_cast(d+j*(n+1))+n,-std::numeric_limits<double>::infinity(),OldMaximum{});
  const auto actual=ugkwp::reduceGasCourantComponents(d,d+n+1,mode==4?nullptr:d+2*(n+1),n,stream);
  double values[]={actual.convective,actual.diffusion,actual.sst};
  for(int j=0;j<3;++j)if(std::memcmp(&expected[j],&values[j],8)){std::cerr<<"mismatch "<<n<<" "<<mode<<" "<<j<<std::endl;return 2;}
  check(cudaFree(d));
 }
 if(streamMode)check(cudaStreamDestroy(stream));
 }
 constexpr int n=65536;double* d;check(cudaMalloc(&d,3*n*sizeof(double)));check(cudaMemset(d,0,3*n*sizeof(double)));
 for(int joint=0;joint<2;++joint){
  auto start=std::chrono::steady_clock::now();double sink=0;
  for(int k=0;k<100;++k){
   if(joint){auto x=ugkwp::reduceGasCourantComponents(d,d+n,d+2*n,n,nullptr);sink+=x.convective+x.diffusion+x.sst;}
   else for(int j=0;j<3;++j)sink+=thrust::reduce(thrust::cuda::par,thrust::device_pointer_cast(d+j*n),thrust::device_pointer_cast(d+(j+1)*n),-std::numeric_limits<double>::infinity(),OldMaximum{});
  }
  std::cout<<(joint?"joint":"separate")<<" us_per_query="<<std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100<<" sum="<<sink<<std::endl;
 }
 check(cudaFree(d));std::cout<<"120 configurations passed, including NaN payloads, mixed signed zero, default/non-default streams"<<std::endl;
}
