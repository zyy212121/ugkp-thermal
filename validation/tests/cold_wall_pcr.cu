#include <cuda_runtime.h>
#include "GpuPrecisionTypes.H"
#include "wall/GpuColdWallThermalCarry.cuh"
#include "wall/GpuColdWall1DAlgebra.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void solve(int n,const double* a,const double* d,const double* c,const double* b,double* x){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*8)return;
 unsigned mask=0xffu<<((threadIdx.x&31)/8*8);
 x[i]=coldWall1DPcrSolve(a[i],d[i],c[i],b[i],threadIdx.x&7,mask);
}
int main(){
 const int count=4096;std::vector<double>a(count*8),d(a.size()),c(a.size()),b(a.size()),x(a.size()),ref(a.size());
 for(int k=0;k<count;++k){double scale=std::pow(10.,(k%101-50)*5.);double edges[9]{};
  for(int j=1;j<8;++j)edges[j]=std::pow(10.,((k+j*7)%11-5)*.4);
  for(int j=0;j<8;++j){int i=k*8+j;a[i]=-scale*edges[j];c[i]=-scale*edges[j+1];d[i]=scale*(.1+edges[j]+edges[j+1]);ref[i]=300.+70.*std::sin(k+j*.9);}
  for(int j=0;j<8;++j){int i=k*8+j;b[i]=d[i]*ref[i]+(j?a[i]*ref[i-1]:0)+(j<7?c[i]*ref[i+1]:0);}
 }
 double *da,*dd,*dc,*db,*dx;size_t bytes=a.size()*sizeof(double);
 check(cudaMalloc(&da,bytes));check(cudaMalloc(&dd,bytes));check(cudaMalloc(&dc,bytes));check(cudaMalloc(&db,bytes));check(cudaMalloc(&dx,bytes));
 check(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(dd,d.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(dc,c.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
 solve<<<count*8/128,128>>>(count,da,dd,dc,db,dx);check(cudaMemcpy(x.data(),dx,bytes,cudaMemcpyDeviceToHost));
 double err=0,res=0;
 for(int k=0;k<count;++k)for(int j=0;j<8;++j){int i=k*8+j;err=fmax(err,std::abs(x[i]-ref[i])/ref[i]);
  double rr=d[i]*x[i]+(j?a[i]*x[i-1]:0)+(j<7?c[i]*x[i+1]:0)-b[i];
  res=fmax(res,std::abs(rr)/(std::abs(d[i]*x[i])+std::abs(b[i])));
  if(!std::isfinite(x[i]))return 1;
 }
 std::printf("4096 heterogeneous SPD systems, scales 1e-250..1e250: relative temperature %.17g, scaled true residual %.17g\n",err,res);
 check(cudaFree(da));check(cudaFree(dd));check(cudaFree(dc));check(cudaFree(db));check(cudaFree(dx));
 return err>1e-10||res>2e-13;
}
