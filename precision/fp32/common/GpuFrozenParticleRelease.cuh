#pragma once
#include "GpuParticleCaloric.H"
// The new zero-speed release changes the active caloric representation. Carry
// profile enthalpy and the saved rebound kinetic reservoir into the free phase;
// do not lose latent heat or the dissipated rebound energy when clearing nodes.
template<class State>
__device__ float frozenReleaseSpecificEnergy(const State& s,int i,int nodes)
{
    float h=0.f;
    for(int n=0;n<nodes;++n)
        h+=(nodes==8?float(s.pColdNodeSpecificEnthalpy[i*8+n])
                    :float(s.pCold2DNodeSpecificEnthalpy[i*64+n]))/nodes;
    const float x=s.puxOld[i],y=s.puyOld[i],z=s.puzOld[i];
    return h+.5f*(x*x+y*y+z*z);
}
template<class State>
__device__ void publishFrozenReleaseEnergy(State& s,int i,float h)
{
    const ugkwp::ParticleCaloric freeLaw;
    s.pT[i]=freeLaw.temperature(h);
}
