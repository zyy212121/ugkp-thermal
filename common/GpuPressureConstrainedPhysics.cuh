#pragma once
#include "GpuPressureParticleUpdate.cuh"
// Constrained particle pressure closure shared by FSH/CHT; scalar precision is an interface type.
template<bool FullMoments, bool CompactParticles = false>
__device__ void accumulatePressureParticleMomentsDevice
(
    const DeviceState& s,
    const int c,
    const int i,
    PressureReal* moments,
    int& count
)
{
    if ((CompactParticles ? s.compactPStatus[i] : s.pStatus[i]) != 1 || (CompactParticles ? s.compactPCellId[i] : s.pCellId[i]) != c)
    {
        return;
    }
    const PressureReal m = clampMin(finiteOr((CompactParticles ? s.compactPm[i] : s.pm[i]), PressureReal(0.0)), PressureReal(0.0));
    const PressureReal ux = finiteOr((CompactParticles ? s.compactPux[i] : s.pux[i]), PressureReal(0.0));
    const PressureReal uy = finiteOr((CompactParticles ? s.compactPuy[i] : s.puy[i]), PressureReal(0.0));
    const PressureReal uz = finiteOr((CompactParticles ? s.compactPuz[i] : s.puz[i]), PressureReal(0.0));
    const PressureReal particleTheta = particleMomentThetaDevice<CompactParticles>(s, i);
    moments[0] += m*ux;
    moments[1] += m*uy;
    moments[2] += m*uz;
    moments[3] +=
        m*(PressureReal(0.5)*sqr3(ux, uy, uz) + PressureReal(1.5)*particleTheta);
    if (FullMoments)
    {
        const PressureReal diameter = clampMin
        (
            finiteOr((CompactParticles ? s.compactPd[i] : s.pd[i]), s.particleDiameterFallback),
            PressureReal(1.0e-12)
        );
        const PressureReal temperature = clampRange
        (
            finiteOr((CompactParticles ? s.compactPT[i] : s.pT[i]), s.TpMin),
            s.TpMin,
            s.TpMax
        );
        moments[4] += m;
        moments[5] += m*diameter;
        moments[6] += m*particleSpecificEnthalpyDevice(temperature);
    }
    ++count;
}

template<bool FullMoments>
__device__ void reducePressureParticleMomentsDevice
(
    PressureReal (&moments)[FullMoments ? 7 : 4],
    int& count,
    PressureReal* warpPartials
)
{
    blockReduceComponentSums<FullMoments ? 7 : 4>(moments, warpPartials);
    __shared__ int countPartials[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31)/32;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        count += __shfl_down_sync(0xffffffffu, count, offset);
    }
    if (lane == 0)
    {
        countPartials[warp] = count;
    }
    __syncthreads();
    if (warp == 0)
    {
        count = lane < warpCount ? countPartials[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            count += __shfl_down_sync(0xffffffffu, count, offset);
        }
    }
    __syncthreads();
}

template<bool FullMoments>
__device__ void storePressureParticleMomentsDevice
(
    DeviceState& s,
    const int c,
    const PressureReal* moments,
    const int count,
    const bool add
)
{
    for (int component = 0; component < (FullMoments ? 7 : 4); ++component)
    {
        PressureReal& output = s.pressureParticleMoments[7u*static_cast<size_t>(c) + component];
        if (add)
        {
            output += moments[component];
        }
        else
        {
            output = moments[component];
        }
    }
    if (add)
    {
        s.pressureParticleCount[c] += count;
    }
    else
    {
        s.pressureParticleCount[c] = count;
    }
}

__device__ int pressureMomentLaneAtRank(unsigned int mask, int rank)
{
    int lane = 0;
    for (int step = 16; step > 0; step >>= 1)
    {
        const unsigned int lowerMask = (1u << step) - 1u;
        const int lowerCount = __popc(mask & lowerMask);
        if (rank >= lowerCount)
        {
            lane += step;
            rank -= lowerCount;
            mask >>= step;
        }
        else
        {
            mask &= lowerMask;
        }
    }
    return lane;
}

template<bool FullMoments>
__device__ void atomicPressureParticleMomentsDevice
(
    DeviceState& s,
    const int c,
    PressureReal* moments,
    int count
)
{
#if __CUDA_ARCH__ >= 700
    const unsigned int activeMask = __activemask();
    const unsigned int groupMask = __match_any_sync(activeMask, c);
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(groupMask) - 1;
    const int rank = __popc(groupMask & ((1u << lane) - 1u));
    const int groupCount = __popc(groupMask);
    const unsigned int shiftedMask = groupMask >> leader;
    const bool contiguous = (shiftedMask & (shiftedMask + 1u)) == 0;
    int offset = 1;
    while (offset < groupCount)
    {
        offset <<= 1;
    }
    for (offset >>= 1; offset > 0; offset >>= 1)
    {
        const bool hasOther = rank + offset < groupCount;
        const int otherLane = hasOther
          ? (contiguous ? lane + offset : pressureMomentLaneAtRank(groupMask, rank + offset))
          : leader;
        for (int component = 0; component < (FullMoments ? 7 : 4); ++component)
        {
            const PressureReal other = __shfl_sync(groupMask, moments[component], otherLane);
            if (hasOther)
            {
                moments[component] += other;
            }
        }
        const int otherCount = __shfl_sync(groupMask, count, otherLane);
        if (hasOther)
        {
            count += otherCount;
        }
    }
    if (lane != leader)
    {
        return;
    }
#endif
    for (int component = 0; component < (FullMoments ? 7 : 4); ++component)
    {
        atomicAdd
        (
            &s.pressureParticleMoments[7u*static_cast<size_t>(c) + component],
            moments[component]
        );
    }
    atomicAdd(&s.pressureParticleCount[c], count);
}

template<bool FullMoments>
__device__ void publishPressureParticleMomentsDevice
(
    DeviceState& s,
    const int c,
    const PressureReal* moments,
    const int count
)
{
    const PressureReal invV = PressureReal(1.0)/clampMin(s.V[c], s.rhoMin);
    s.momRhoUPx[c] = moments[0]*invV;
    s.momRhoUPy[c] = moments[1]*invV;
    s.momRhoUPz[c] = moments[2]*invV;
    s.momRhoEP[c] = moments[3]*invV;
    if (FullMoments)
    {
        s.momRhoP[c] = moments[4]*invV;
        s.momRhoPD[c] = moments[5]*invV;
        s.momRhoHpP[c] = moments[6]*invV;
    }
    s.cellParticleCount[c] = count;
    if (c == 0)
    {
        s.cellParticleCount[s.nCells] = 0;
    }
    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        s.epsS[c] = PressureReal(0.0);
        s.rhoUsx[c] = PressureReal(0.0);
        s.rhoUsy[c] = PressureReal(0.0);
        s.rhoUsz[c] = PressureReal(0.0);
        s.rhoEs[c] = PressureReal(0.0);
        s.Usx[c] = PressureReal(0.0);
        s.Usy[c] = PressureReal(0.0);
        s.Usz[c] = PressureReal(0.0);
        s.theta[c] = PressureReal(0.0);
        return;
    }
    const PressureReal totalMomX = finiteOr(s.momRhoUPx[c], PressureReal(0.0));
    const PressureReal totalMomY = finiteOr(s.momRhoUPy[c], PressureReal(0.0));
    const PressureReal totalMomZ = finiteOr(s.momRhoUPz[c], PressureReal(0.0));
    PressureReal totalEnergy = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0));
    const PressureReal kinetic =
        PressureReal(0.5)*sqr3(totalMomX, totalMomY, totalMomZ)/rhoP;
    if (totalEnergy < kinetic)
    {
        totalEnergy = kinetic;
        s.momRhoEP[c] = totalEnergy;
    }
    s.epsS[c] = rhoP/s.rhoSolid;
    s.rhoUsx[c] = totalMomX;
    s.rhoUsy[c] = totalMomY;
    s.rhoUsz[c] = totalMomZ;
    s.rhoEs[c] = totalEnergy;
    s.Usx[c] = totalMomX/rhoP;
    s.Usy[c] = totalMomY/rhoP;
    s.Usz[c] = totalMomZ/rhoP;
    s.theta[c] = clampMin
    (
        (totalEnergy - kinetic)/(PressureReal(1.5)*rhoP),
        PressureReal(0.0)
    );
}

template<bool FullMoments>
__global__ void publishPressureParticleMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    publishPressureParticleMomentsDevice<FullMoments>
    (
        s,
        c,
        s.pressureParticleMoments + 7u*static_cast<size_t>(c),
        s.pressureParticleCount[c]
    );
}

template<bool FullMoments, bool CompactParticles = false>
__device__ void accumulatePressureParticleMomentsAtomicDevice
(
    DeviceState& s,
    const int c,
    const int i
)
{
    PressureReal moments[FullMoments ? 7 : 4] = {};
    int count = 0;
    accumulatePressureParticleMomentsDevice<FullMoments, CompactParticles>(s, c, i, moments, count);
    if (count != 0)
    {
        atomicPressureParticleMomentsDevice<FullMoments>(s, c, moments, count);
    }
}

template<bool CompactParticles = false>
__device__ void applyPressureParticleStateDevice
(
    DeviceState& s,
    const int i,
    const PressureReal ux0,
    const PressureReal uy0,
    const PressureReal uz0,
    const PressureReal ux1,
    const PressureReal uy1,
    const PressureReal uz1,
    const PressureReal theta1,
    const PressureReal thermalScale,
    const PressureReal thetaScale,
    const bool resolved
)
{
    if ((CompactParticles ? s.compactPStuck[i] : s.pStuck[i]) != 0)
    {
        (CompactParticles ? s.compactPux[i] : s.pux[i]) = PressureReal(0.0);
        (CompactParticles ? s.compactPuy[i] : s.puy[i]) = PressureReal(0.0);
        (CompactParticles ? s.compactPuz[i] : s.puz[i]) = PressureReal(0.0);
        if ((CompactParticles ? s.compactPStuck[i] : s.pStuck[i]) == Foam::gpuThermal::particleWallDeposited)
        {
            s.puxOld[CompactParticles ? s.sortedParticleIndex[i] : i] = PressureReal(0.0);
            s.puyOld[CompactParticles ? s.sortedParticleIndex[i] : i] = PressureReal(0.0);
            s.puzOld[CompactParticles ? s.sortedParticleIndex[i] : i] = PressureReal(0.0);
        }
        return;
    }
    updateMobilePressureParticle<CompactParticles>
        (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
}

#include "GpuPressureKickAccumulation.cuh"
template<bool FullMoments>
struct ConstrainedPressureScratch
{
    __device__ static __forceinline__ void initialise(DeviceState& s,int c)
    {
        for(int component=0;component<(FullMoments?7:4);++component)
            s.pressureParticleMoments[7u*static_cast<size_t>(c)+component]=PressureReal(0);
        s.pressureParticleCount[c]=0;
    }
};
template<bool FullMoments>
__global__ void accumulateCollisionalPressureKickByCellKernel
(DeviceState* sp,const PressureTime kickDt,const int computeScale,const int initialiseMoments)
{
    runPressureKickAccumulation<ConstrainedPressureScratch<FullMoments>>
        (sp,kickDt,computeScale,initialiseMoments);
}

#include "GpuPressureLimiter.cuh"
