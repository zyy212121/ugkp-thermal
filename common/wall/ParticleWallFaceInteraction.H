#pragma once

template<class State>
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
unsigned char particleWallFaceInteraction(const State& s,int face)
{
    return s.particleStuckCandidateMask ? s.particleStuckCandidateMask[face] : 0;
}
