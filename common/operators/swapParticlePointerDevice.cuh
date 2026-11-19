#pragma once

template<class T>
__host__ __device__ __forceinline__ void swapParticlePointerDevice(T*& active, T*& scratch)
{
    T* const oldActive = active;
    active = scratch;
    scratch = oldActive;
}
