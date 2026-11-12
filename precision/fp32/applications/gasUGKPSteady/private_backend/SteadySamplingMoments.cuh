#pragma once
#include "../gpu/GpuSteadySampling.H"
#include "GpuPrecisionTypes.H"
#include <cmath>
#include <limits>

// Same stored conserved quantities as the resident gas/parcel moment pipeline.
// Gas entries are intrinsic FV storage; porosity source exchanges are reported
// separately from gas boundary flux, never mislabeled a closed mixture budget.
template<class State>
__device__ GpuReal steadySampleValue(const State& s,int c,int k)
{
    switch(k)
    {
        case 0:return s.rho[c]; case 1:return s.rhoUx[c];
        case 2:return s.rhoUy[c]; case 3:return s.rhoUz[c];
        case 4:return s.rhoE[c];
        case 5:return s.sstConfigured?s.rhoK[c]:0;
        case 6:return s.sstConfigured?s.rhoOmega[c]:0;
        case 7:return s.epsS[c]*s.rhoSolid;
        case 8:return s.rhoUsx[c]; case 9:return s.rhoUsy[c];
        case 10:return s.rhoUsz[c]; case 11:return s.rhoEs[c];
        case 12:return s.rhoHp[c]; case 13:return s.rhoDs[c];
        default:return ugkwp::mixtureGasActive(s)?s.gasSpecies.rho[(k-14)*s.nCells+c]:0;
    }
}

template<class State>
__global__ void beginSteadySampleWindowKernel(State* sp)
{
    State& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    for(int k=0;k<GpuSteadySamplingReport::fields;++k)
    {
        const int j=k*s.nCells+c;
        s.steadySampleSum[j]=0;s.steadySampleSquare[j]=0;
        // Center moments on this window's initial represented value so a large
        // formation-energy baseline cannot erase physical RMS fluctuations.
        s.steadySampleScratch[j]=steadySampleValue(s,c,k);
        s.steadySampleBegin[j]=steadySampleValue(s,c,k)*((k<7||k>=14)?1-s.epsS[c]:1);
    }
}

template<class State>
__global__ void accumulateSteadySampleWindowKernel(State* sp,GpuTime dt)
{
    State& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    for(int k=0;k<GpuSteadySamplingReport::fields;++k)
    {
        const int j=k*s.nCells+c;const GpuReal q=steadySampleValue(s,c,k);
        const GpuReal delta=q-s.steadySampleScratch[j];
        s.steadySampleSum[j]+=GpuReal(dt)*delta;s.steadySampleSquare[j]+=GpuReal(dt)*delta*delta;
    }
}

template<class State>
__global__ void finishSteadySampleWindowKernel(State* sp,GpuTime duration)
{
    State& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    const int n=GpuSteadySamplingReport::fields*s.nCells;
    for(int k=0;k<GpuSteadySamplingReport::fields;++k)
    {
        const int j=k*s.nCells+c;
        const GpuReal meanDelta=s.steadySampleSum[j]/GpuReal(duration);
        const GpuReal mean=s.steadySampleScratch[j]+meanDelta;
        const GpuReal previous=s.steadySampleMean[j];
        GpuReal characteristic=0;
        if(k>=1&&k<=3)characteristic=std::sqrt(std::fabs(s.rho[c]*s.p[c]));
        if(k>=8&&k<=10)characteristic=std::sqrt(std::fabs(s.epsS[c]*s.rhoSolid*s.rhoEs[c]));
        if(k==4)characteristic=std::fabs(s.p[c])
            +GpuReal(0.5)*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
        const GpuReal scale=std::fmax(std::numeric_limits<GpuReal>::min(),k==4?characteristic:
            std::fmax(characteristic,std::fmax(std::fabs(mean),std::fabs(previous))));
        s.steadySampleScratch[j]=std::fabs(mean-previous)/scale;
        const GpuReal variance=std::fmax(GpuReal(0),s.steadySampleSquare[j]/GpuReal(duration)-meanDelta*meanDelta);
        // RMS measures stochastic variability; it is not a standard error or
        // independent-sample confidence interval, and does not gate stationarity.
        s.steadySampleScratch[n+j]=std::sqrt(variance)/scale;
        s.steadySampleScratch[2*n+j]=s.steadySampleBegin[j]*s.V[c];
        s.steadySampleScratch[3*n+j]=steadySampleValue(s,c,k)*s.V[c]*((k<7||k>=14)?1-s.epsS[c]:1);
        s.steadySampleScratch[4*n+j]=scale*s.V[c]*((k<7||k>=14)?1-s.epsS[c]:1);
        s.steadySampleMean[j]=mean;
    }
}
