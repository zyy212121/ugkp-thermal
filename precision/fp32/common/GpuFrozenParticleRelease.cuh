#pragma once
#include "wall/GpuColdWall1DHandles.cuh"
#include "GpuParticleCaloric.H"



template<class State>
__device__ float frozenReleaseSpecificEnergy(const State& s,int i,int nodes)
{
    float h=0.f;
    for(int n=0;n<nodes;++n)
        h+=(nodes==8?float(s.pColdNodeSpecificEnthalpy[Foam::gpuWall::coldWallSlot(s,i)*8+n])
                    :float(s.pCold2DNodeSpecificEnthalpy[i*64+n]))/nodes;
    const float x=s.puxOld[i],y=s.puyOld[i],z=s.puzOld[i];
    return h+.5f*(x*x+y*y+z*z);
}
template<class State>
__device__ void publishFrozenReleaseEnergy(State& s,int i,float h,bool cold1D=false)
{
    (void)cold1D;
    const ugkwp::ParticleCaloric freeLaw;
    s.pT[i]=freeLaw.temperature(h);
}
