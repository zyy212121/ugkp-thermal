#pragma once
#include "wall/GpuColdWall1DHandles.cuh"
#include "GpuParticleCaloric.H"



template<class State>
__device__ auto frozenReleaseCarry(const State& s,int i,int)
    -> decltype(double(s.pThermalEnthalpyResidual[i]))
{return double(s.pThermalEnthalpyResidual[i]);}
template<class State>
__device__ double frozenReleaseCarry(const State&,int,long){return 0.;}
template<class State>
__device__ auto storeFrozenReleaseCarry(State& s,int i,double value,int)
    -> decltype(s.pThermalEnthalpyResidual[i]=value,void())
{s.pThermalEnthalpyResidual[i]=value;}
template<class State>
__device__ void storeFrozenReleaseCarry(State&,int,double,long){}
template<class State>
__device__ double frozenReleaseSpecificEnergy(const State& s,int i,int nodes)
{
    double h=0.;
    for(int n=0;n<nodes;++n)
        h+=(nodes==8?double(s.pColdNodeSpecificEnthalpy[Foam::gpuWall::coldWallSlot(s,i)*8+n])
                    :double(s.pCold2DNodeSpecificEnthalpy[i*64+n]))/nodes;
    const double x=s.puxOld[i],y=s.puyOld[i],z=s.puzOld[i];
    return h+(nodes==8?0.:frozenReleaseCarry(s,i,0))+.5*(x*x+y*y+z*z);
}
template<class State>
__device__ void publishFrozenReleaseEnergy(State& s,int i,double h,bool cold1D=false)
{
    const ugkwp::ParticleCaloric freeLaw;
    s.pT[i]=freeLaw.temperature(h);
    storeFrozenReleaseCarry(s,i,cold1D?0.:h-freeLaw.enthalpy(double(s.pT[i])),0);
}
