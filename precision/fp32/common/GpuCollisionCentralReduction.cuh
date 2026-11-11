#pragma once
#include "GpuCollisionCentralMoment.H"
// Shuffle the complete state before merging it. A component-wise sum is not
// a reduction operation for a mean and its central energy.
template<int N>
__device__ void blockReduceCollisionMoments(float (&q)[N], float* scratch)
{
    const unsigned mask=__activemask();
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const int warps=(blockDim.x+31)/32;
    const int width=min(32,int(blockDim.x)-32*warp);
    for(int offset=16;offset;offset>>=1)
    {
        float other[N];
        #pragma unroll
        for(int k=0;k<N;++k)other[k]=__shfl_down_sync(mask,q[k],offset);
        if(lane+offset<width)mergeCollisionMomentGroup(q,other);
    }
    if(!lane)for(int k=0;k<N;++k)scratch[k*warps+warp]=q[k];
    __syncthreads();
    if(!warp)
    {
        for(int k=0;k<N;++k)q[k]=lane<warps?scratch[k*warps+lane]:0.0f;
        for(int offset=16;offset;offset>>=1)
        {
            float other[N];
            #pragma unroll
            for(int k=0;k<N;++k)other[k]=__shfl_down_sync(mask,q[k],offset);
            if(lane+offset<min(32,int(blockDim.x)))mergeCollisionMomentGroup(q,other);
        }
    }
}
