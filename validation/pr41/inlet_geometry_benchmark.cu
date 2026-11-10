// Compare with the parent revision by compiling this same source with
// -DUGKP_LEGACY_INLET_BUILDER and that revision's header first in -I.
#include "BoundaryInjectionSampling.H"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define BENCH_HD __host__ __device__
#else
#define BENCH_HD
#endif
using ugkp::gpu::geometry::Point;
using ugkwp::BoundaryInjectionTriangle;
BENCH_HD int buildOne(int i,BoundaryInjectionTriangle* out){
    const double warp=.01+1e-5*(i%19);
    const Point p[4]={{-warp,0,0},{warp,0,1},{-warp,1,1},{warp,1,0}};
    const int ids[4]={0,1,2,3};int scratch[4];
#ifdef UGKP_LEGACY_INLET_BUILDER
    return ugkwp::buildBoundaryInjectionTriangles(p,ids,4,scratch,out,{1e-6,0,0});
#else
    return ugkwp::buildBoundaryInjectionTriangles(p,ids,4,scratch,out,
        {-warp/3,.5,.5},{-1,0,0},{1e-6,0,0});
#endif
}
BENCH_HD Point sampleOne(int i){
    const BoundaryInjectionTriangle t[2]={
        {{0,0,0},{2,0,0},{0,1,0},1},
        {{10,0,0},{6,0,0},{0,1,0},4}};
    Point p;
    ugkwp::sampleBoundaryInjection(t,0,2,static_cast<unsigned long long>(i)+123,p.x,p.y,p.z);
    return p;
}
#ifdef __CUDACC__
__global__ void builds(BoundaryInjectionTriangle* out,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)buildOne(i,out+4*i);
}
__global__ void samples(Point* out,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=sampleOne(i);
}
#endif
int main(){
#ifdef UGKP_LEGACY_INLET_BUILDER
    const char* version="legacy";
#else
    const char* version="projected";
#endif
    constexpr int faces=65536,particles=262144;
#ifdef __CUDACC__
    BoundaryInjectionTriangle* triangles;Point* positions;
    assert(cudaMalloc(&triangles,faces*4*sizeof(*triangles))==cudaSuccess);
    assert(cudaMalloc(&positions,particles*sizeof(*positions))==cudaSuccess);
    for(int i=0;i<10;++i){builds<<<(faces+127)/128,128>>>(triangles,faces);samples<<<(particles+255)/256,256>>>(positions,particles);}
    assert(cudaDeviceSynchronize()==cudaSuccess);
    cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
    for(int operation=0;operation<2;++operation){
        double observations[9];const int repeats=operation?200:50;
        for(double& value:observations){
            cudaEventRecord(begin);
            for(int k=0;k<repeats;++k){
                if(operation)samples<<<(particles+255)/256,256>>>(positions,particles);
                else builds<<<(faces+127)/128,128>>>(triangles,faces);
            }
            cudaEventRecord(end);assert(cudaEventSynchronize(end)==cudaSuccess);
            float ms;cudaEventElapsedTime(&ms,begin,end);
            value=ms/repeats/(operation?particles:faces)*1e6;
        }
        std::sort(observations,observations+9);
        printf("BENCH %s CUDA %s ms_per_million=%.6f min=%.6f max=%.6f\n",version,operation?"sample":"initial_build",observations[4],observations[0],observations[8]);
    }
    cudaEventDestroy(begin);cudaEventDestroy(end);cudaFree(triangles);cudaFree(positions);
#else
    std::vector<BoundaryInjectionTriangle> triangles(faces*4);std::vector<Point> positions(particles);
    volatile double checksum=0;
    for(int operation=0;operation<2;++operation){
        double observations[7];constexpr int repeats=4;
        for(double& value:observations){
            const auto begin=std::chrono::steady_clock::now();
            for(int k=0;k<repeats;++k){
                if(operation){
                    for(int i=0;i<particles;++i)positions[i]=sampleOne(i);
                    checksum=checksum+positions[k].x;
                }else{
                    for(int i=0;i<faces;++i)assert(buildOne(i,triangles.data()+4*i)==2);
                    checksum=checksum+triangles[4*k+1].cumulativeArea;
                }
            }
            const auto end=std::chrono::steady_clock::now();
            value=std::chrono::duration<double,std::milli>(end-begin).count()/repeats/(operation?particles:faces)*1e6;
        }
        std::sort(observations,observations+7);
        printf("BENCH %s CPU %s ms_per_million=%.6f min=%.6f max=%.6f\n",version,operation?"sample":"initial_build",observations[3],observations[0],observations[6]);
    }
    printf("checksum=%.17g\n",double(checksum));
#endif
}
