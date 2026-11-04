#pragma once
#include "GpuParticleCaloric.H"
// The new zero-speed release changes the active caloric representation. Carry
// profile enthalpy and the saved rebound kinetic reservoir into the free phase;
// do not lose latent heat or the dissipated rebound energy when clearing nodes.
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
        h+=(nodes==8?double(s.pColdNodeSpecificEnthalpy[i*8+n])
                    :double(s.pCold2DNodeSpecificEnthalpy[i*64+n]))/nodes;
    const double x=s.puxOld[i],y=s.puyOld[i],z=s.puzOld[i];
    return h+frozenReleaseCarry(s,i,0)+.5*(x*x+y*y+z*z);
}
template<class State>
__device__ void publishFrozenReleaseEnergy(State& s,int i,double h)
{
    const ugkwp::ParticleCaloric freeLaw;
    s.pT[i]=freeLaw.temperature(h);
    storeFrozenReleaseCarry(s,i,h-freeLaw.enthalpy(double(s.pT[i])),0);
}
