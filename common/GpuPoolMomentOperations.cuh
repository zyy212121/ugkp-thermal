#pragma once
enum class PoolReductionTopology { sharedTree, warpComponents };
template<PoolReductionTopology Topology>
__device__ __forceinline__ void reducePoolMoments
(GPU_OPERATOR_REAL (&sums)[8], GPU_OPERATOR_REAL* scratch)
{
    if constexpr (Topology == PoolReductionTopology::warpComponents)
        blockReduceComponentSums<8>(sums, scratch);
    else
    {
        #pragma unroll
        for (int k=0; k<8; ++k) scratch[k*blockDim.x + threadIdx.x] = sums[k];
        __syncthreads();
        for (int stride=blockDim.x>>1; stride>0; stride>>=1)
        {
            if (threadIdx.x < stride)
            {
                #pragma unroll
                for (int k=0; k<8; ++k)
                    scratch[k*blockDim.x + threadIdx.x] += scratch[k*blockDim.x + threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int k=0; k<8; ++k) sums[k] = scratch[k*blockDim.x];
        }
    }
}
template<bool Add>
__device__ __forceinline__ void publishPoolCell
(DeviceState& s, const int c, const GPU_OPERATOR_REAL (&sums)[8])
{
    if constexpr (Add)
    {
        s.poissonPoolMass[c] += sums[0];
        s.poissonPoolMomX[c] += sums[1];
        s.poissonPoolMomY[c] += sums[2];
        s.poissonPoolMomZ[c] += sums[3];
        s.poissonPoolEnergy[c] += sums[4];
        s.poissonPoolDiameter[c] += sums[5];
        s.poissonPoolDiameter2[c] += sums[6];
        s.poolThermalCount[c] += static_cast<int>(sums[7]);
    }
    else
    {
        s.poissonPoolMass[c] = sums[0];
        s.poissonPoolMomX[c] = sums[1];
        s.poissonPoolMomY[c] = sums[2];
        s.poissonPoolMomZ[c] = sums[3];
        s.poissonPoolEnergy[c] = sums[4];
        s.poissonPoolDiameter[c] = sums[5];
        s.poissonPoolDiameter2[c] = sums[6];
        s.poolThermalCount[c] = static_cast<int>(sums[7]);
    }
}
__device__ __forceinline__ void publishPoolPartial
(DeviceState& s, const int task, const GPU_OPERATOR_REAL (&sums)[8])
{
    #pragma unroll
    for (int k=0; k<8; ++k) s.csrHeavyPartials[8u*static_cast<size_t>(task)+k] = sums[k];
}
__device__ __forceinline__ void zeroPoolPartial(DeviceState& s, const int task)
{
    const GPU_OPERATOR_REAL zero[8] = {};
    publishPoolPartial(s, task, zero);
}
