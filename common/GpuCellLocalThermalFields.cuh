#pragma once
#include "GpuParticleFields.cuh"
#include "operators/swapParticlePointerDevice.cuh"
// Shared full thermal payload. Compile-time layout adapters preserve the
// existing cold1D copy form; there is no runtime policy or extra traversal.
template<bool HasContactAge, bool PackedCold1D>
struct CellLocalThermalExtraFields
{
    template<class State>
    static __host__ __device__ __forceinline__ void swap(State& s)
    {
#define GPU_SWAP_FIELD(src, dest, unused) swapParticlePointerDevice(s.src, s.dest);
        if constexpr (HasContactAge)
        {
            GPU_PARTICLE_FIELDS_SEPARATE_AGE(GPU_SWAP_FIELD)
        }
        GPU_PARTICLE_FIELDS_CONTACT_SCALAR(GPU_SWAP_FIELD)
        if (s.coldWallSolidificationEnabled != 0)
        {
            GPU_PARTICLE_FIELDS_COLD1D_ARRAY(GPU_SWAP_FIELD)
            GPU_PARTICLE_FIELDS_COLD1D_SCALAR(GPU_SWAP_FIELD)
        }
        if (s.coldWall2DEnabled != 0)
        {
            GPU_PARTICLE_FIELDS_COLD2D_ARRAY(GPU_SWAP_FIELD)
            GPU_PARTICLE_FIELDS_COLD2D_SCALAR(GPU_SWAP_FIELD)
        }
#undef GPU_SWAP_FIELD
    }
    template<int Width, bool Packed, class Scalar>
    static __device__ __forceinline__ void copyArray
    (const Scalar* src, Scalar* dest, const int i, const int dst)
    {
        if constexpr (Packed)
        {
            static_assert(Width % 4 == 0, "cold-wall stride must preserve float4 alignment");
            static_assert(sizeof(Scalar) == sizeof(float), "packed cold-wall storage must be float");
            #pragma unroll
            for (int j = 0; j < Width/4; ++j)
                reinterpret_cast<float4*>(dest)[dst*(Width/4)+j] =
                    reinterpret_cast<const float4*>(src)[i*(Width/4)+j];
        }
        else
        {
            for (int j = 0; j < Width; ++j) dest[dst*Width+j] = src[i*Width+j];
        }
    }
    template<class State>
    static __device__ __forceinline__ void copyContactAge(State& s, const int i, const int dst)
    {
#define GPU_COPY_AGE(src, dest, value) s.dest[dst] = value;
        if constexpr (HasContactAge)
        {
            GPU_PARTICLE_FIELDS_SEPARATE_AGE(GPU_COPY_AGE)
        }
#undef GPU_COPY_AGE
    }
    template<class State>
    static __device__ __forceinline__ void copy(State& s, const int i, const int dst)
    {
#define GPU_COPY_SCALAR(src, dest, value) s.dest[dst] = value;
        GPU_PARTICLE_FIELDS_CONTACT_SCALAR(GPU_COPY_SCALAR)
        if (s.coldWallSolidificationEnabled != 0)
        {
#define GPU_COPY_ARRAY(src, dest, width) \
            if constexpr (PackedCold1D) copyArray<width, true>(s.src, s.dest, i, dst); \
            else { for (int j = 0; j < width; ++j) s.dest[dst*width+j] = s.src[i*width+j]; }
            GPU_PARTICLE_FIELDS_COLD1D_ARRAY(GPU_COPY_ARRAY)
#undef GPU_COPY_ARRAY
            GPU_PARTICLE_FIELDS_COLD1D_SCALAR(GPU_COPY_SCALAR)
        }
        if (s.coldWall2DEnabled != 0)
        {
#define GPU_COPY_ARRAY(src, dest, width) \
            for (int j = 0; j < width; ++j) s.dest[dst*width+j] = s.src[i*width+j];
            GPU_PARTICLE_FIELDS_COLD2D_ARRAY(GPU_COPY_ARRAY)
#undef GPU_COPY_ARRAY
            GPU_PARTICLE_FIELDS_COLD2D_SCALAR(GPU_COPY_SCALAR)
        }
#undef GPU_COPY_SCALAR
    }
    template<class State>
    static __device__ __forceinline__ void publish(State& s, const int dst)
    {
        if (s.compactPStuck[dst] != Foam::gpuThermal::particleWallMobile)
            Foam::gpuWall::publishWallBoundParticleIndex(s, dst);
    }
};
