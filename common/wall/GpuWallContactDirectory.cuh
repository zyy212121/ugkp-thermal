#ifndef UGKP_UNIFIED_WALL_CONTACT_DIRECTORY_CUH
#define UGKP_UNIFIED_WALL_CONTACT_DIRECTORY_CUH

                                                                     
                                                                             
                                                             

namespace Foam
{
namespace gpuWall
{

template<class State>
__device__ inline void publishWallBoundParticleIndex
(
    State& state,
    const int particleIndex
)
{
    const int slot = atomicAdd(state.wallBoundParticleCountDevice, 1);
    if
    (
        slot < 0
     || slot >= state.particleCapacity
     || particleIndex < 0
     || particleIndex >= state.particleCapacity
    )
    {
        asm("trap;");
    }
    state.wallBoundParticleIndex[slot] = particleIndex;
}

template<class State>
__device__ inline int wallBoundDirectoryCount(const State& state)
{
    return max(0, min(*state.wallBoundParticleCountDevice, state.particleCapacity));
}

template<class State>
__device__ inline int wallBoundDirectoryParticle
(
    const State& state,
    const int entry
)
{
    if (entry < 0 || entry >= wallBoundDirectoryCount(state))
    {
        return -1;
    }
    const int particleIndex = state.wallBoundParticleIndex[entry];
    return
    (
        particleIndex >= 0 && particleIndex < state.particleCapacity
    ) ? particleIndex : -1;
}

}                         
}                      

#endif
