#include "common/GpuGasCourantReduction.cuh"
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>
struct ReferenceMaximum {
 __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
void check(cudaError_t e){assert(e==cudaSuccess);}
int main(){
 for(int streamMode=0;streamMode<2;++streamMode){
 cudaStream_t stream=nullptr;if(streamMode)check(cudaStreamCreate(&stream));
 for(int n:{0,1,31,32,127,128,255,256,1025,65537})for(int mode=0;mode<5;++mode){
  std::vector<double> h(n+1);
  for(int i=0;i<n;++i)h[i]=(i*13)%101-50.;
  if(mode==1)for(auto& x:h)x=-0.;
  if(mode==2&&n)h[n/2]=INFINITY;
  if(mode==3&&n){h[n/3]=nan("11");h[n/2]=nan("29");}
  if(mode==4)for(int i=0;i<n;++i)h[i]=i%2?0.:-0.;
  double*d=nullptr;check(cudaMalloc(&d,h.size()*sizeof(double)));check(cudaMemcpy(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice));
  const double expected=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(d),thrust::device_pointer_cast(d)+n,-std::numeric_limits<double>::infinity(),ReferenceMaximum{});
  const double actual=ugkwp::reduceGasAcousticCourant(d,n,stream);
  if(std::memcmp(&expected,&actual,sizeof(double))){std::cerr<<"mismatch "<<n<<" "<<mode<<std::endl;return 2;}
  check(cudaFree(d));
 }
 if(streamMode)check(cudaStreamDestroy(stream));
 }
 std::cout<<"100 scalar reduction configurations passed"<<std::endl;
}
