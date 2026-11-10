#include "BoundaryInjectionSampling.H"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <algorithm>
using ugkwp::BoundaryInjectionTriangle;
struct Position { double x,y,z; };
__host__ __device__ Position draw(const BoundaryInjectionTriangle* triangles, int i) {
    Position p{};
    ugkwp::sampleBoundaryInjection(triangles,0,2,
        ugkwp::injectionPositionSeed(123456,8992,.28000000001,i),p.x,p.y,p.z);
    return p;
}
__global__ void samples(const BoundaryInjectionTriangle* triangles, Position* out,int n,bool reverse) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { if(reverse)i=n-1-i; out[i]=draw(triangles,i); }
}
int main() {
    constexpr int n=262144;
    const BoundaryInjectionTriangle triangles[2]={
        {{0,0,0},{2,0,0},{0,1,0},1},
        {{10,0,0},{6,0,0},{0,1,0},4}};
    BoundaryInjectionTriangle* device;Position *a,*b;
    assert(cudaMalloc(&device,sizeof(triangles))==cudaSuccess);
    assert(cudaMalloc(&a,n*sizeof(Position))==cudaSuccess);
    assert(cudaMalloc(&b,n*sizeof(Position))==cudaSuccess);
    assert(cudaMemcpy(device,triangles,sizeof(triangles),cudaMemcpyHostToDevice)==cudaSuccess);
    samples<<<(n+127)/128,128>>>(device,a,n,false);
    samples<<<(n+255)/256,256>>>(device,b,n,true);
    assert(cudaDeviceSynchronize()==cudaSuccess);
    std::vector<Position> actual(n),reversed(n);
    assert(cudaMemcpy(actual.data(),a,n*sizeof(Position),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(reversed.data(),b,n*sizeof(Position),cudaMemcpyDeviceToHost)==cudaSuccess);
    int second=0;double mx=0,my=0,maxError=0;
    for(int i=0;i<n;++i) {
        const auto p=actual[i],q=draw(triangles,i),v=reversed[i];
        assert(p.x==v.x&&p.y==v.y&&p.z==v.z);
        maxError=std::max(maxError,std::max(std::abs(p.x-q.x),std::abs(p.y-q.y)));
        assert(p.y>0&&p.y<1&&p.z==0);
        if(p.x>9) {++second;assert((p.x-10)/6+p.y<1);} else assert(p.x/2+p.y<1);
        mx+=p.x;my+=p.y;
    }
    mx/=n;my/=n;
    assert(std::abs(double(second)/n-.75)<.005);
    assert(std::abs(mx-(.25*2/3.+.75*12))<.05);
    assert(std::abs(my-1/3.)<.005);
    assert(maxError<5e-15);
    assert(ugkwp::injectionPositionSeed(1,2,.2,3)!=ugkwp::injectionPositionSeed(1,2,.2000000000001,3));
    assert(ugkwp::injectionPositionSeed(1,2,.2,3)!=ugkwp::injectionPositionSeed(1,3,.2,3));
    double x=4,y=5,z=6;assert(!ugkwp::sampleBoundaryInjection(nullptr,0,0,1,x,y,z));
    assert(x==4&&y==5&&z==6);
    cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int k=0;k<100;++k)samples<<<(n+255)/256,256>>>(device,a,n,false);
    cudaEventRecord(stop);cudaEventSynchronize(stop);float ms;cudaEventElapsedTime(&ms,start,stop);
    printf("PASS n=%d area_fraction=%.9f mean=(%.9f,%.9f) CPU_GPU_max=%.3g repeat=bitwise milliseconds_per_million=%.6f\n",n,double(second)/n,mx,my,maxError,ms/100/n*1e6);
    cudaFree(device);cudaFree(a);cudaFree(b);cudaEventDestroy(start);cudaEventDestroy(stop);
}
