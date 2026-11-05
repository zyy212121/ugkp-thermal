// The same admissible profile must have the same limiter at every scale.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#endif
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) (x)
constexpr double OfSmall=2.220446049250313e-16;
#include "operators/updateBarthLimiter.cuh"
__device__ int checkLimiters()
{
    for (int exponent=0; exponent<=280; exponent+=20)
    {
        const double scale=pow(10.0,-exponent);
        for (int sign=-1; sign<=1; sign+=2)
        {
            const double centre=sign>0?2*scale:4*scale;
            const double predicted=sign>0?-3*scale:9*scale;
            double barth=1,venkat=1;
            updateBarthLimiter(centre,predicted,0,6*scale,barth);
            updateVenkatakrishnanLimiter(centre,predicted,0,6*scale,venkat);
            if (fabs(barth-.4)>1e-14 || fabs(venkat-.375)>1e-14) return 1;
        }
        double flat=.7,atMinimum=1;
        updateVenkatakrishnanLimiter(scale,scale,0,2*scale,flat);
        updateVenkatakrishnanLimiter(0,-scale,0,scale,atMinimum);
        if(flat!=.7 || atMinimum!=0) return 2;
    }
    double largeRatio=1;
    updateVenkatakrishnanLimiter(0,1e-280,-1,1e280,largeRatio);
    return largeRatio==1 ? 0:3;
}
#ifdef __CUDACC__
__global__ void probe(int* result){*result=checkLimiters();}
#endif
int main()
{
    int result=0;
#ifdef __CUDACC__
    int* device=nullptr;
    if(cudaMalloc(&device,sizeof(int))!=cudaSuccess)return 10;
    probe<<<1,1>>>(device);
    if(cudaMemcpy(&result,device,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 11;
    if(cudaFree(device)!=cudaSuccess)return 12;
#else
    result=checkLimiters();
#endif
    std::printf("scale-invariant limiter: %s (%d)\n",result?"FAIL":"PASS",result);
    return result;
}
