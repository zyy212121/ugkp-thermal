#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class T>
__device__ void swapParticlePointerDevice(T*& active, T*& scratch)
{
    T* const oldActive = active;
    active = scratch;
    scratch = oldActive;
}
