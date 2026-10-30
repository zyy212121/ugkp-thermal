// CUDA arithmetic regression for the shared density/flux inventory limiter.
#include "gasTransport/PositiveInventory.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cassert>
struct Case {double density,volume,dt,outflow;};
__global__ void evaluate(const Case* cases,double* scale,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){auto q=cases[i];
 scale[i]=ugkwp::inventoryOutflowScale(q.density,q.volume,q.dt,q.outflow);}
}
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
int main(){
 const int exponents[]={-1074,-1060,-1022,-900,-500,-100,0,100,500,900,1023};
 std::vector<Case> cases;
 for(int d:exponents)for(int v:exponents)for(int t:exponents)for(int f:exponents)
 cases.push_back({std::ldexp(1.,d),std::ldexp(1.,v),std::ldexp(1.,t),std::ldexp(1.,f)});
 cases.push_back({0,3.6691894773708457e-7,9.6873727665352311e-9,1.1644331826788377e-317});
 Case* input;double* output;check(cudaMalloc(&input,cases.size()*sizeof(Case)));check(cudaMalloc(&output,cases.size()*sizeof(double)));
 check(cudaMemcpy(input,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice));
 evaluate<<<(cases.size()+127)/128,128>>>(input,output,int(cases.size()));check(cudaGetLastError());
 std::vector<double> scales(cases.size());check(cudaMemcpy(scales.data(),output,scales.size()*sizeof(double),cudaMemcpyDeviceToHost));
 const long double tiny=std::numeric_limits<double>::denorm_min();
 for(size_t i=0;i<cases.size();++i){const auto c=cases[i];const double q=scales[i];
  const long double ratio=(static_cast<long double>(c.density)*c.volume)/(static_cast<long double>(c.dt)*c.outflow);
  assert(std::isfinite(q)&&q>=0&&q<=1);
  if(ratio>=1)assert(q==1);
  else {assert(static_cast<long double>(q)<=ratio);const long double expected=.999L*ratio;
   assert(std::abs(static_cast<long double>(q)-expected)<=8*std::numeric_limits<double>::epsilon()*expected+2*tiny);}
 }
 const auto c=cases.back();assert(c.dt*c.outflow==0 && (c.dt/c.volume)*c.outflow>0 && scales.back()==0);
 std::printf("{\"cases\":%zu,\"zero_donor_scale\":%.17g,\"counterexample_unlimited_density\":%.17g}\n",cases.size(),scales.back(),-(c.dt/c.volume)*c.outflow);
 check(cudaFree(input));check(cudaFree(output));
}
