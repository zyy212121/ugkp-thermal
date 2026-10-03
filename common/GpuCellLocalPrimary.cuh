#pragma once
#include "GpuParticleFields.cuh"
#ifdef GPU_PARTICLE_EXTRA_FIELDS
using ParticleFieldExtension = GPU_PARTICLE_EXTRA_FIELDS;
#else
struct ParticleFieldExtension
{
    template<class State>
    static __host__ __device__ __forceinline__ void swap(State&) {}
};
#endif
// Shared primary fields with optional compile-time thermal extension.
__device__ __forceinline__ void copyCellLocalParticle
(DeviceState& s, const int i, const int c, const int dst)
{
#define GPU_COPY_SCALAR(src, dest, value) s.dest[dst] = value;
    GPU_PARTICLE_FIELDS_PRIMARY_BEFORE_CONTACT(GPU_COPY_SCALAR)
#ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::copyContactAge(s, i, dst);
#endif
    GPU_PARTICLE_FIELDS_PRIMARY_AFTER_CONTACT(GPU_COPY_SCALAR)
#ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::copy(s, i, dst);
#endif
    GPU_PARTICLE_FIELDS_IDENTITY(GPU_COPY_SCALAR)
#undef GPU_COPY_SCALAR
#ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::publish(s, dst);
#endif
}

#ifdef GPU_PARTICLE_EXTRA_FIELDS
#undef GPU_PARTICLE_EXTRA_FIELDS
#endif
