#include "GpuResidentStrict.cu"
#include <cstdio>
#include <new>
#include <cstdlib>
#include <cmath>
#include <cstring>
using Real=GPU_OPERATOR_REAL;
__device__ GpuReal baseline_solidEpsFromMomentDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return GPU_R(0.0);
    }

    const GpuReal rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_R(0.0)), GPU_R(0.0));

    return clampRange
    (
        rhoP/clampMin(s.rhoSolid, GPU_TINY(1.0e-300)),
        GPU_R(0.0),
        GPU_R(1.0)
    );
}
__device__ GpuReal baseline_solidPressureFromMomentsDevice
(
    const DeviceState& s,
    const int c
)
{
    if (c < 0 || c >= s.nCells)
    {
        return GPU_R(0.0);
    }

    const GpuReal rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_R(0.0)), GPU_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return GPU_R(0.0);
    }

    const GpuReal eps =
        clampRange(rhoP/clampMin(s.rhoSolid, GPU_TINY(1.0e-300)), GPU_R(0.0), GPU_R(1.0));
    const GpuReal ux = finiteOr(s.momRhoUPx[c], GPU_R(0.0))/rhoP;
    const GpuReal uy = finiteOr(s.momRhoUPy[c], GPU_R(0.0))/rhoP;
    const GpuReal uz = finiteOr(s.momRhoUPz[c], GPU_R(0.0))/rhoP;
    const GpuReal kinetic = GPU_R(0.5)*rhoP*sqr3(ux, uy, uz);
    const GpuReal theta =
        clampMin
        (
            (finiteOr(s.momRhoEP[c], kinetic) - kinetic)/(GPU_R(1.5)*rhoP),
            GPU_R(0.0)
        );
    GpuReal pColl = GPU_R(0.0);
    if (s.collisionalPressureEnabled)
    {
        const GpuReal g0 = radialDistributionG0Device(eps);
        pColl = ugkwp::collisionalPressure
        (
            s.collisionalRestitution,
            s.rhoSolid,
            eps,
            g0,
            theta
        );
        pColl = clampRange(finiteOr(pColl, GPU_R(0.0)), GPU_R(0.0), OfGreat);
    }

    return clampRange(finiteOr(pColl, OfGreat), GPU_R(0.0), OfGreat);
}
__device__ GpuReal baseline_granularCollisionTauFromCellDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return OfGreat;
    }

    const GpuReal rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_R(0.0)), GPU_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return OfGreat;
    }

    const GpuReal eps =
        clampRange(rhoP/clampMin(s.rhoSolid, GPU_TINY(1.0e-300)), GPU_R(0.0), GPU_R(1.0));

    const GpuReal usx = finiteOr(s.momRhoUPx[c], GPU_R(0.0))/rhoP;
    const GpuReal usy = finiteOr(s.momRhoUPy[c], GPU_R(0.0))/rhoP;
    const GpuReal usz = finiteOr(s.momRhoUPz[c], GPU_R(0.0))/rhoP;

    const GpuReal e =
        clampMin(finiteOr(s.momRhoEP[c], GPU_R(0.0)), GPU_R(0.0))/rhoP;

    const GpuReal theta =
        clampMin((e - GPU_R(0.5)*sqr3(usx, usy, usz))/GPU_R(1.5), GPU_R(0.0));

    const GpuReal dPart =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/rhoP, s.particleDiameterFallback),
            GPU_R(1.0e-12)
        );

    const GpuReal g0 = radialDistributionG0Device(eps);
    const GpuReal lmfp = ugkwp::granularMeanFreePath
    (
        OfPi,
        dPart,
        eps,
        g0,
        OfSmall
    );

    return clampMin
    (
        ugkwp::granularCollisionTime(lmfp, theta, OfSmall),
        OfSmall
    );
}
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GasPrimDevice baseline_riemannFacePrimitiveForGradient
(
    const DeviceState& s,
    const int c,
    const int f
)
{
    GasPrimDevice centre = makeGasPrimDevice
    (
        s.rho[c], s.Ux[c], s.Uy[c], s.Uz[c], s.p[c],
        s.Rgas, s.rhoMin, s.TgasMin
    );
    centre.T = clampMin(finiteOr(s.Tgas[c], centre.T), s.TgasMin);
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const int other = isPeriodicFace(s, f) ? nei : (c == own ? nei : own);
        if (other < 0 || other >= s.nCells)
        {
            return centre;
        }
        GasPrimDevice adjacent = makeGasPrimDevice
        (
            s.rho[other], s.Ux[other], s.Uy[other], s.Uz[other], s.p[other],
            s.Rgas, s.rhoMin, s.TgasMin
        );
        adjacent.T = clampMin
        (
            finiteOr(s.Tgas[other], adjacent.T),
            s.TgasMin
        );
        const GPU_OPERATOR_REAL ownerWeight = clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        const GPU_OPERATOR_REAL wc = c == own ? ownerWeight : GPU_OPERATOR_R(1.0) - ownerWeight;
        GasPrimDevice face = makeGasPrimDevice
        (
            wc*centre.rho + (GPU_OPERATOR_R(1.0) - wc)*adjacent.rho,
            wc*centre.ux + (GPU_OPERATOR_R(1.0) - wc)*adjacent.ux,
            wc*centre.uy + (GPU_OPERATOR_R(1.0) - wc)*adjacent.uy,
            wc*centre.uz + (GPU_OPERATOR_R(1.0) - wc)*adjacent.uz,
            wc*centre.p + (GPU_OPERATOR_R(1.0) - wc)*adjacent.p,
            s.Rgas, s.rhoMin, s.TgasMin
        );
        face.T = wc*centre.T + (GPU_OPERATOR_R(1.0) - wc)*adjacent.T;
        return face;
    }

    const int kind = s.riemannBoundaryKind[f];
    if (kind == 4 || kind == 3)
    {
        return centre;
    }
    if (kind == 1)
    {
        const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
        const GPU_OPERATOR_REAL nx = s.Sfx[f]/area;
        const GPU_OPERATOR_REAL ny = s.Sfy[f]/area;
        const GPU_OPERATOR_REAL nz = s.Sfz[f]/area;
        const GPU_OPERATOR_REAL un = centre.ux*nx + centre.uy*ny + centre.uz*nz;
        GasPrimDevice face = centre;
        face.ux -= un*nx;
        face.uy -= un*ny;
        face.uz -= un*nz;
        if (s.riemannBoundaryTFix[f] != 0)
        {
            face.T = clampMin(finiteOr(s.riemannBoundaryT[f], centre.T), s.TgasMin);
            face.rho = face.p/clampMin(s.Rgas*face.T, OfSmall);
        }
        return face;
    }

    if (kind == 2)
    {
        GasPrimDevice wall = centre;
        wall.ux = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        wall.uy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        wall.uz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        if (s.riemannBoundaryTFix[f] != 0)
        {
            wall.T = clampMin
            (
                finiteOr(s.riemannBoundaryT[f], centre.T),
                s.TgasMin
            );
            wall.rho = wall.p/clampMin(s.Rgas*wall.T, OfSmall);
        }
        return wall;
    }

    return riemannBoundaryState(s, f, centre);
}

__device__ void baseline_atomicAddParticleWallEnergyByFace
(
    DeviceState& s,
    GpuWallEnergy* const wallEnergyLedger,
    const int globalFaceId,
    const GpuReal wallEnergyJ
)
{
    if (wallEnergyJ == GPU_R(0.0))
    {
        return;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned int active = __activemask();
    const int ledgerChannel =
        wallEnergyLedger == s.particleWallDepositedEnergy ? 0 : 1;
    const int faceChannelKey = 2*globalFaceId + ledgerChannel;
    const unsigned int group =
        __match_any_sync(active, faceChannelKey);
    const int leader = __ffs(static_cast<int>(group)) - 1;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    GpuWallEnergy groupEnergy = 0.0;
    unsigned int remaining = group;
    while (remaining != 0u)
    {
        const int sourceLane = __ffs(static_cast<int>(remaining)) - 1;
        groupEnergy += static_cast<GpuWallEnergy>(__shfl_sync(group, wallEnergyJ, sourceLane));
        remaining &= remaining - 1u;
    }
    if (lane == leader)
    {
        atomicAdd(&wallEnergyLedger[globalFaceId], groupEnergy);
    }
#else
    atomicAdd(&wallEnergyLedger[globalFaceId], static_cast<GpuWallEnergy>(wallEnergyJ));
#endif
}
#include "GpuPrecisionTypes.H"
#ifndef BASELINE_GPU_DRAG_ALGEBRA_CUH
#define BASELINE_GPU_DRAG_ALGEBRA_CUH

#include <cmath>

#if defined(__CUDACC__)
#define UGKWP_DRAG_HD __host__ __device__ __forceinline__
#else
#define UGKWP_DRAG_HD inline
#endif

namespace baselineDrag
{

UGKWP_DRAG_HD GpuReal schillerNaumannCorrection(const GpuReal reynolds)
{
    return GPU_R(1.0) + GPU_R(0.15)*pow(reynolds, GPU_R(0.687));
}
UGKWP_DRAG_HD GpuReal gidaspowCdRe
(const GpuReal alpha, const GpuReal reynolds, const GpuReal residualRe)
{
    if (alpha >= GPU_R(0.8))
    {
        const GpuReal alphaRe = alpha*reynolds;
        const GpuReal cdRe = alphaRe < GPU_R(1000.0)
            ? GPU_R(24.0)*schillerNaumannCorrection(alphaRe)
            : GPU_R(0.44)*fmax(alphaRe, residualRe);
        return cdRe*pow(alpha, -GPU_R(2.65));
    }
    return (GPU_R(4.0)/GPU_R(3.0))
        *(GPU_R(150.0)*(GPU_R(1.0)-alpha)/alpha + GPU_R(1.75)*reynolds);
}

UGKWP_DRAG_HD GpuReal gasUgkpReynolds
(
    const GpuReal gasDensity,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal gasViscosity
)
{
    return
        gasDensity*diameter*relativeSpeed
       /fmax(gasViscosity, GPU_R(1.0e-30));
}

UGKWP_DRAG_HD GpuReal gasUgkpSchillerNaumannCoefficient
(
    const GpuReal reynolds
)
{
    const GpuReal reSafe = fmax(reynolds, GPU_R(1.0e-12));
    if (reSafe < GPU_R(1000.0))
    {
        return GPU_R(24.0)/reSafe*schillerNaumannCorrection(reSafe);
    }
    return GPU_R(0.44);
}

UGKWP_DRAG_HD GpuReal gasUgkpSchillerNaumannInverseResponseTime
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal solidDensity,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal denominatorRegularization
)
{
    const GpuReal reynolds = gasUgkpReynolds
    (
        gasDensity,
        diameter,
        relativeSpeed,
        gasViscosity
    );
    const GpuReal coefficient =
        gasUgkpSchillerNaumannCoefficient(reynolds);
    return
        GPU_R(0.75)*coefficient*gasDensity*relativeSpeed
       /(solidDensity*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowCdRe
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal gasVolumeFraction,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal residualRe
)
{
    const GpuReal alphaGas =
        fmin(fmax(gasVolumeFraction, GPU_R(1.0e-12)), GPU_R(1.0));
    const GpuReal reynolds = fmax
    (
        gasUgkpReynolds
        (
            gasDensity,
            diameter,
            relativeSpeed,
            gasViscosity
        ),
        GPU_R(0.0)
    );
    return gidaspowCdRe(alphaGas, reynolds, residualRe);
}

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowInverseResponseTime
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal gasVolumeFraction,
    const GpuReal solidDensity,
    const GpuReal diameterInput,
    const GpuReal relativeSpeed,
    const GpuReal denominatorRegularization,
    const GpuReal residualRe
)
{
    const GpuReal diameter = fmax(diameterInput, GPU_R(1.0e-30));
    return
        GPU_R(0.75)
       *gasUgkpGidaspowCdRe
        (
            gasDensity,
            gasViscosity,
            gasVolumeFraction,
            diameterInput,
            relativeSpeed,
            residualRe
        )
       *fmax(gasViscosity, GPU_R(1.0e-30))
       /(solidDensity*diameter*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD GpuReal fshChtSchillerNaumannInverseRelaxationTime
(
    const GpuReal gasDensityInput,
    const GpuReal gasViscosity,
    const GpuReal solidDensityInput,
    const GpuReal diameterInput,
    const GpuReal relativeSpeedInput
)
{
    const GpuReal mu = fmax(gasViscosity, GPU_R(1.0e-30));
    const GpuReal re =
        fmax(gasDensityInput, GPU_R(0.0))*fmax(diameterInput, GPU_R(1.0e-30))
       *fmax(relativeSpeedInput, GPU_R(0.0))/mu;
    if (re <= GPU_R(1.0e-30) || relativeSpeedInput <= GPU_R(1.0e-30))
    {
        return GPU_R(0.0);
    }
    const GpuReal coefficient =
        re < GPU_R(1000.0)
      ? GPU_R(24.0)*schillerNaumannCorrection(re)/re
      : GPU_R(0.44);
    return
        GPU_R(0.75)*coefficient*fmax(gasDensityInput, GPU_R(0.0))
       *fmax(relativeSpeedInput, GPU_R(0.0))
       /(fmax(solidDensityInput, GPU_R(1.0e-30))
        *fmax(diameterInput, GPU_R(1.0e-30)));
}

UGKWP_DRAG_HD GpuReal fshChtGidaspowInverseRelaxationTime
(
    const GpuReal gasDensityInput,
    const GpuReal gasVolumeFraction,
    const GpuReal gasViscosity,
    const GpuReal solidDensityInput,
    const GpuReal diameterInput,
    const GpuReal relativeSpeedInput,
    const GpuReal residualRe
)
{
    const GpuReal alpha =
        fmin(fmax(gasVolumeFraction, GPU_R(1.0e-12)), GPU_R(1.0));
    const GpuReal mu = fmax(gasViscosity, GPU_R(1.0e-30));
    const GpuReal diameter = fmax(diameterInput, GPU_R(1.0e-30));
    const GpuReal re =
        fmax(gasDensityInput, GPU_R(0.0))*diameter
       *fmax(relativeSpeedInput, GPU_R(0.0))/mu;
    const GpuReal cdRe = gidaspowCdRe(alpha, re, residualRe);
    return
        GPU_R(0.75)*cdRe*mu
       /(fmax(solidDensityInput, GPU_R(1.0e-30))*diameter*diameter);
}

}

#undef UGKWP_DRAG_HD

#endif

// Shared unsorted thermal projection; preserves the published-state reconstruction.
__global__ void baselinePressureCell
(
    DeviceState* sp,
    const PressureTime kickDt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    PressureReal dpx = PressureReal(0.0);
    PressureReal dpy = PressureReal(0.0);
    PressureReal dpz = PressureReal(0.0);
    PressureReal de = PressureReal(0.0);
    const int startFace = s.cellPlaneStart[c];
    const int faceCount = s.cellPlaneCount[c];
    for (int j = 0; j < faceCount; ++j)
    {
        const int f = s.cellFaceId[startFace + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const PressureReal sign = s.faceOwner[f] == c ? PressureReal(1.0) : -PressureReal(1.0);
        dpx -= sign*s.solidPressurePhiMomX[f];
        dpy -= sign*s.solidPressurePhiMomY[f];
        dpz -= sign*s.solidPressurePhiMomZ[f];
        de -= sign*s.solidPressurePhiEnergy[f];
    }

    const PressureReal factor = kickDt/clampMin(s.V[c], OfVSmall);
    dpx = finiteOr(factor*dpx, PressureReal(0.0));
    dpy = finiteOr(factor*dpy, PressureReal(0.0));
    dpz = finiteOr(factor*dpz, PressureReal(0.0));
    de = finiteOr(factor*de, PressureReal(0.0));
    s.pressureDeltaMomX[c] = dpx;
    s.pressureDeltaMomY[c] = dpy;
    s.pressureDeltaMomZ[c] = dpz;
    s.pressureDeltaEnergy[c] = de;

    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const PressureReal px1 = finiteOr(s.momRhoUPx[c], PressureReal(0.0)) + dpx;
    const PressureReal py1 = finiteOr(s.momRhoUPy[c], PressureReal(0.0)) + dpy;
    const PressureReal pz1 = finiteOr(s.momRhoUPz[c], PressureReal(0.0)) + dpz;
    const PressureReal e1 = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0)) + de;
    const PressureReal theta1 =
        clampMin
        (
            pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(PressureReal(1.5)*rhoP),
            PressureReal(0.0)
        );

    s.momRhoUPx[c] = px1;
    s.momRhoUPy[c] = py1;
    s.momRhoUPz[c] = pz1;
    s.momRhoEP[c] = e1;
    s.rhoUsx[c] = px1;
    s.rhoUsy[c] = py1;
    s.rhoUsz[c] = pz1;
    s.rhoEs[c] = e1;
    s.Usx[c] = px1/rhoP;
    s.Usy[c] = py1/rhoP;
    s.Usz[c] = pz1/rhoP;
    s.theta[c] = theta1;
}


template<bool FullMoments>
__global__ void baselinePressureParticles
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] == 0)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        if (s.pStuck[i] != 0)
        {
            s.pux[i] = PressureReal(0.0);
            s.puy[i] = PressureReal(0.0);
            s.puz[i] = PressureReal(0.0);
            if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
            {
                s.puxOld[i] = PressureReal(0.0);
                s.puyOld[i] = PressureReal(0.0);
                s.puzOld[i] = PressureReal(0.0);
            }
            accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
            continue;
        }

        const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
        if (rhoP <= s.epsSMin*s.rhoSolid)
        {
            accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
            continue;
        }
        const PressureReal dpx = finiteOr(s.pressureDeltaMomX[c], PressureReal(0.0));
        const PressureReal dpy = finiteOr(s.pressureDeltaMomY[c], PressureReal(0.0));
        const PressureReal dpz = finiteOr(s.pressureDeltaMomZ[c], PressureReal(0.0));
        const PressureReal de = finiteOr(s.pressureDeltaEnergy[c], PressureReal(0.0));
        const PressureReal px1 = finiteOr(s.momRhoUPx[c], PressureReal(0.0));
        const PressureReal py1 = finiteOr(s.momRhoUPy[c], PressureReal(0.0));
        const PressureReal pz1 = finiteOr(s.momRhoUPz[c], PressureReal(0.0));
        const PressureReal e1 = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0));
        const PressureReal px0 = px1 - dpx;
        const PressureReal py0 = py1 - dpy;
        const PressureReal pz0 = pz1 - dpz;
        const PressureReal e0 = e1 - de;
        const PressureReal ux0 = px0/rhoP;
        const PressureReal uy0 = py0/rhoP;
        const PressureReal uz0 = pz0/rhoP;
        const PressureReal ux1 = px1/rhoP;
        const PressureReal uy1 = py1/rhoP;
        const PressureReal uz1 = pz1/rhoP;
        const PressureReal theta0 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)
               /(PressureReal(1.5)*rhoP),
                PressureReal(0.0)
            );
        const PressureReal theta1 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)
               /(PressureReal(1.5)*rhoP),
                PressureReal(0.0)
            );
        const bool resolved = theta0 > PressureReal(10.0)*s.thetaMin;
        const PressureReal thermalScale =
            resolved ? sqrt(clampMin(theta1/theta0, PressureReal(0.0))) : PressureReal(0.0);
        const PressureReal thetaScale = resolved ? thermalScale*thermalScale : PressureReal(0.0);
        updateMobilePressureParticle<false>
            (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
        accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
    }
}


template<class T>void mem(T*&p){if(cudaMallocManaged(&p,1024*sizeof(T))!=cudaSuccess)abort();for(int i=0;i<1024;i++)::new(static_cast<void*>(p+i)) T{};}
void deviceSync(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){printf("FAIL CUDA %s\n",cudaGetErrorString(e));exit(10);}}
void ck(bool x,const char*m){if(!x){printf("FAIL %s\n",m);exit(11);}}
void equal(double a,double b,const char*m){if(std::memcmp(&a,&b,sizeof(double))){printf("FAIL bits %s %.17g != %.17g\n",m,a,b);exit(12);}}
__global__ void primitives(DeviceState*s,double*values){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=512)return;
 int c=i%18-1; int f=i%40;int fc=s->faceOwner[f];if(f<s->nInternalFaces && (i&1))fc=s->faceNeighbour[f];
 Real cur[]={solidEpsFromMomentDevice(*s,c),solidPressureFromMomentsDevice(*s,c),granularCollisionTauFromCellDevice(*s,c)};
 Real old[]={baseline_solidEpsFromMomentDevice(*s,c),baseline_solidPressureFromMomentsDevice(*s,c),baseline_granularCollisionTauFromCellDevice(*s,c)};
 auto face=riemannFacePrimitiveForGradient(*s,fc,f);auto bf=baseline_riemannFacePrimitiveForGradient(*s,fc,f);
 Real faces[]={face.rho,face.ux,face.uy,face.uz,face.p,face.T},refs[]={bf.rho,bf.ux,bf.uy,bf.uz,bf.p,bf.T};
 for(int j=0;j<3;j++){values[32*i+2*j]=cur[j];values[32*i+2*j+1]=old[j];}
 for(int j=0;j<6;j++){values[32*i+6+2*j]=faces[j];values[32*i+7+2*j]=refs[j];}
}
__global__ void dragRates(double*values){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=512)return;
 const Real slips[]={Real(0),Real(1e-32),Real(1e-12),Real(.1),Real(999.999),Real(1000),Real(1000.001),Real(-1)};
 const Real densities[]={Real(0),Real(1),Real(-1),Real(1e-30)};
 const Real diameters[]={Real(1),Real(1e-4),Real(1e-31),Real(0)};
 const Real alphas[]={Real(.799999),Real(.8),Real(.800001),Real(1)};
 Real slip=slips[i%8],rho=densities[(i/8)%4],d=diameters[(i/32)%4],alpha=alphas[(i/128)%4],mu=Real(1),rs=Real(3000),reg=Real(1e-30),rr=Real(1e-12);
 Real cur[]={ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannInverseResponseTime(rho,mu,rs,d,slip,reg),ugkwpGpuDragAlgebra::gasUgkpGidaspowInverseResponseTime(rho,mu,alpha,rs,d,slip,reg,rr),ugkwpGpuDragAlgebra::fshChtSchillerNaumannInverseRelaxationTime(rho,mu,rs,d,slip),ugkwpGpuDragAlgebra::fshChtGidaspowInverseRelaxationTime(rho,alpha,mu,rs,d,slip,rr)};
 Real old[]={baselineDrag::gasUgkpSchillerNaumannInverseResponseTime(rho,mu,rs,d,slip,reg),baselineDrag::gasUgkpGidaspowInverseResponseTime(rho,mu,alpha,rs,d,slip,reg,rr),baselineDrag::fshChtSchillerNaumannInverseRelaxationTime(rho,mu,rs,d,slip),baselineDrag::fshChtGidaspowInverseRelaxationTime(rho,alpha,mu,rs,d,slip,rr)};
 for(int j=0;j<4;j++){values[8*i+2*j]=cur[j];values[8*i+2*j+1]=old[j];}
}

__global__ void wallChannels(DeviceState*s,DeviceState*r){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=129)return;int f=i%3;Real e=(i%7==0)?Real(0):Real((i%2?-1:1)*(1+i%5)*.125);auto a=i%2?s->particleWallReflectedEnergy:s->particleWallDepositedEnergy;auto b=i%2?r->particleWallReflectedEnergy:r->particleWallDepositedEnergy;atomicAddParticleWallEnergyByFace(*s,a,f,e);baseline_atomicAddParticleWallEnergyByFace(*r,b,f,e);}
// The scalar Thomas solve is a separate path from the eight-lane PCR solve.
// Start both from the same stored profile values at every step; do not compare two
// independently drifting trajectories or freeze the corrected implementation.
__global__ void coldScalar(DeviceState*r,double*out,double dt,double gasConductance,double duration,double gasTemperature){
 if(threadIdx.x!=0)return;
 Real h[8],ring[8],mean=0;for(int n=0;n<8;n++){h[n]=Real(r->pColdNodeSpecificEnthalpy[n]);ring[n]=Real(r->pColdRingSolidMass[n]);mean+=h[n]/Real(8);}
 GpuTime age=r->pColdContactAge[0];Real frozen=Real(r->pColdFrozenArea[0]),gasIncrement=0;
 bool gasValid=Foam::gpuThermal::coldWallGasSpecificEnthalpyIncrement(mean,Real(3.2e-9),Real(gasTemperature),Real(gasConductance),GpuTime(dt),r->coldWallSolidificationParameters,gasIncrement);
 auto b=Foam::gpuThermal::advanceColdWallProfile(h,ring,age,frozen,r->coldWallSolidificationParameters,
  Real(1e-12),Real(3.2e-9),Real(2.2e-8),Real(1.7e-8),GpuTime(duration),Real(.42),GpuTime(dt),Real(300),Real(13000),Real(.1),Real(gasTemperature),Real(gasConductance));
 out[1]=b.valid;out[3]=b.meanTemperatureK;out[5]=b.frozenFootprintAreaM2;out[7]=b.wallEnergyJ;out[9]=gasValid;out[10]=double(Real(3.2e-9))*gasIncrement;
 if(b.valid){for(int n=0;n<8;n++){r->pColdNodeSpecificEnthalpy[n]=h[n];r->pColdRingSolidMass[n]=ring[n];}r->pColdContactAge[0]=age;r->pColdFrozenArea[0]=frozen;}
}
__global__ void coldPair(DeviceState*s,double*out,int stable,double dt,double gasConductance,double duration=8e-6,double gasTemperature=3200){int lane=threadIdx.x;Real ma=0,fa=0,ea=0;bool failA=false;
 const Real volume=Real(1e-12),mass=Real(3.2e-9),area=Real(2.2e-8),contact=Real(1.7e-8);
 bool a;
 a=advanceColdWall1DThermalGroup(*s,0,lane,0xffu,volume,mass,area,contact,GpuTime(duration),Real(.42),GpuTime(dt),Real(300),Real(13000),Real(.1),Real(gasTemperature),Real(gasConductance),ma,fa,ea);
 if(lane==0){out[0]=a;out[2]=ma;out[4]=fa;out[6]=ea;out[8]=failA;}}
void seedColdReference(const DeviceState*s,DeviceState*r){
 for(int n=0;n<8;n++){r->pColdNodeSpecificEnthalpy[n]=s->pColdNodeSpecificEnthalpy[n];r->pColdRingSolidMass[n]=s->pColdRingSolidMass[n];}
 r->pColdContactAge[0]=s->pColdContactAge[0];r->pColdFrozenArea[0]=s->pColdFrozenArea[0];
}
template<class Storage> double coldStorageUlp(Storage x){return fmax(fabs(double(std::nextafter(x,Storage(INFINITY)))-x),fabs(double(x)-std::nextafter(x,Storage(-INFINITY))));}
void nearCold(double actual,double expected,double bound,const char*name){
 if(!std::isfinite(actual)||!std::isfinite(expected)||fabs(actual-expected)>bound){printf("FAIL %s actual=%.17g expected=%.17g bound=%.17g\n",name,actual,expected,bound);exit(13);}
}

int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s,*reference;mem(s);mem(reference);
mem(s->csrMaximumOccupancy);mem(reference->csrMaximumOccupancy);
mem(s->csrHeavyTaskCount);mem(reference->csrHeavyTaskCount);
mem(s->csrHeavyCellCount);mem(reference->csrHeavyCellCount);
mem(s->csrHeavyTaskCursor);mem(reference->csrHeavyTaskCursor);
mem(s->particleCountDevice);mem(reference->particleCountDevice);
mem(s->faceOwner);mem(reference->faceOwner);
mem(s->faceNeighbour);mem(reference->faceNeighbour);
mem(s->facePeriodicPair);mem(reference->facePeriodicPair);
mem(s->facePeriodicDx);mem(reference->facePeriodicDx);
mem(s->facePeriodicDy);mem(reference->facePeriodicDy);
mem(s->facePeriodicDz);mem(reference->facePeriodicDz);
mem(s->V);mem(reference->V);
mem(s->Cx);mem(reference->Cx);
mem(s->Cy);mem(reference->Cy);
mem(s->Cz);mem(reference->Cz);
mem(s->faceCx);mem(reference->faceCx);
mem(s->faceCy);mem(reference->faceCy);
mem(s->faceCz);mem(reference->faceCz);
mem(s->Sfx);mem(reference->Sfx);
mem(s->Sfy);mem(reference->Sfy);
mem(s->Sfz);mem(reference->Sfz);
mem(s->magSf);mem(reference->magSf);
mem(s->deltaCoeffs);mem(reference->deltaCoeffs);
mem(s->faceWeight);mem(reference->faceWeight);
mem(s->cellLength);mem(reference->cellLength);
mem(s->cellPlaneStart);mem(reference->cellPlaneStart);
mem(s->cellPlaneCount);mem(reference->cellPlaneCount);
mem(s->cellFaceId);mem(reference->cellFaceId);
mem(s->cellFaceNeighbor);mem(reference->cellFaceNeighbor);
mem(s->cellFaceKind);mem(reference->cellFaceKind);
mem(s->cellFaceRestitution);mem(reference->cellFaceRestitution);
mem(s->cellFaceTangential);mem(reference->cellFaceTangential);
mem(s->planeNx);mem(reference->planeNx);
mem(s->planeNy);mem(reference->planeNy);
mem(s->planeNz);mem(reference->planeNz);
mem(s->planeD);mem(reference->planeD);
mem(s->rho);mem(reference->rho);
mem(s->rhoUx);mem(reference->rhoUx);
mem(s->rhoUy);mem(reference->rhoUy);
mem(s->rhoUz);mem(reference->rhoUz);
mem(s->rhoE);mem(reference->rhoE);
mem(s->rhoNext);mem(reference->rhoNext);
mem(s->rhoUxNext);mem(reference->rhoUxNext);
mem(s->rhoUyNext);mem(reference->rhoUyNext);
mem(s->rhoUzNext);mem(reference->rhoUzNext);
mem(s->rhoENext);mem(reference->rhoENext);
mem(s->gasPhiRho);mem(reference->gasPhiRho);
mem(s->gasPhiRhoUx);mem(reference->gasPhiRhoUx);
mem(s->gasPhiRhoUy);mem(reference->gasPhiRhoUy);
mem(s->gasPhiRhoUz);mem(reference->gasPhiRhoUz);
mem(s->gasPhiRhoE);mem(reference->gasPhiRhoE);
mem(s->gasFluxPositivityScale);mem(reference->gasFluxPositivityScale);
mem(s->gasHllcAdcSensor);mem(reference->gasHllcAdcSensor);
mem(s->particleStuckCandidateMask);mem(reference->particleStuckCandidateMask);
mem(s->particleWallDepositedEnergy);mem(reference->particleWallDepositedEnergy);
mem(s->particleWallReflectedEnergy);mem(reference->particleWallReflectedEnergy);
mem(s->particleWallRepresentedContactArea);mem(reference->particleWallRepresentedContactArea);
mem(s->particleWallContactAreaScale);mem(reference->particleWallContactAreaScale);
mem(s->particleWallEffusivityByFace);mem(reference->particleWallEffusivityByFace);
mem(s->finiteContactRateTable);mem(reference->finiteContactRateTable);
mem(s->gasBoundaryKind);mem(reference->gasBoundaryKind);
mem(s->gasBoundaryRhoFix);mem(reference->gasBoundaryRhoFix);
mem(s->gasBoundaryUFix);mem(reference->gasBoundaryUFix);
mem(s->gasBoundaryPFix);mem(reference->gasBoundaryPFix);
mem(s->gasBoundaryTFix);mem(reference->gasBoundaryTFix);
mem(s->gasBoundaryPWave);mem(reference->gasBoundaryPWave);
mem(s->gasBoundaryPWaveGamma);mem(reference->gasBoundaryPWaveGamma);
mem(s->gasBoundaryPWaveFieldInf);mem(reference->gasBoundaryPWaveFieldInf);
mem(s->gasBoundaryPWaveLInf);mem(reference->gasBoundaryPWaveLInf);
mem(s->gasBoundaryRho);mem(reference->gasBoundaryRho);
mem(s->gasBoundaryUx);mem(reference->gasBoundaryUx);
mem(s->gasBoundaryUy);mem(reference->gasBoundaryUy);
mem(s->gasBoundaryUz);mem(reference->gasBoundaryUz);
mem(s->gasBoundaryP);mem(reference->gasBoundaryP);
mem(s->gasBoundaryT);mem(reference->gasBoundaryT);
mem(s->riemannBoundaryKind);mem(reference->riemannBoundaryKind);
mem(s->riemannBoundaryRhoFix);mem(reference->riemannBoundaryRhoFix);
mem(s->riemannBoundaryUFix);mem(reference->riemannBoundaryUFix);
mem(s->riemannBoundaryPFix);mem(reference->riemannBoundaryPFix);
mem(s->riemannBoundaryTFix);mem(reference->riemannBoundaryTFix);
mem(s->riemannBoundaryPWave);mem(reference->riemannBoundaryPWave);
mem(s->riemannBoundaryPWaveGamma);mem(reference->riemannBoundaryPWaveGamma);
mem(s->riemannBoundaryPWaveFieldInf);mem(reference->riemannBoundaryPWaveFieldInf);
mem(s->riemannBoundaryPWaveLInf);mem(reference->riemannBoundaryPWaveLInf);
mem(s->riemannBoundaryRho);mem(reference->riemannBoundaryRho);
mem(s->riemannBoundaryUx);mem(reference->riemannBoundaryUx);
mem(s->riemannBoundaryUy);mem(reference->riemannBoundaryUy);
mem(s->riemannBoundaryUz);mem(reference->riemannBoundaryUz);
mem(s->riemannBoundaryP);mem(reference->riemannBoundaryP);
mem(s->riemannBoundaryT);mem(reference->riemannBoundaryT);
mem(s->scheduledInletFaceMask);mem(reference->scheduledInletFaceMask);
mem(s->pressureScheduleTimes);mem(reference->pressureScheduleTimes);
mem(s->pressureScheduleValues);mem(reference->pressureScheduleValues);
mem(s->volumeFractionScheduleTimes);mem(reference->volumeFractionScheduleTimes);
mem(s->volumeFractionScheduleValues);mem(reference->volumeFractionScheduleValues);
mem(s->gradPx);mem(reference->gradPx);
mem(s->gradPy);mem(reference->gradPy);
mem(s->gradPz);mem(reference->gradPz);
mem(s->gradRhoX);mem(reference->gradRhoX);
mem(s->gradRhoY);mem(reference->gradRhoY);
mem(s->gradRhoZ);mem(reference->gradRhoZ);
mem(s->gradUxX);mem(reference->gradUxX);
mem(s->gradUxY);mem(reference->gradUxY);
mem(s->gradUxZ);mem(reference->gradUxZ);
mem(s->gradUyX);mem(reference->gradUyX);
mem(s->gradUyY);mem(reference->gradUyY);
mem(s->gradUyZ);mem(reference->gradUyZ);
mem(s->gradUzX);mem(reference->gradUzX);
mem(s->gradUzY);mem(reference->gradUzY);
mem(s->gradUzZ);mem(reference->gradUzZ);
mem(s->gradTX);mem(reference->gradTX);
mem(s->gradTY);mem(reference->gradTY);
mem(s->gradTZ);mem(reference->gradTZ);
mem(s->gasGradientLimiterRho);mem(reference->gasGradientLimiterRho);
mem(s->gasGradientLimiterUx);mem(reference->gasGradientLimiterUx);
mem(s->gasGradientLimiterUy);mem(reference->gasGradientLimiterUy);
mem(s->gasGradientLimiterUz);mem(reference->gasGradientLimiterUz);
mem(s->gasGradientLimiterP);mem(reference->gasGradientLimiterP);
mem(s->gasGradientLimiterT);mem(reference->gasGradientLimiterT);
mem(s->nut);mem(reference->nut);
mem(s->gasDiffusionNumber);mem(reference->gasDiffusionNumber);
mem(s->rhoK);mem(reference->rhoK);
mem(s->rhoOmega);mem(reference->rhoOmega);
mem(s->rhoKInitial);mem(reference->rhoKInitial);
mem(s->rhoOmegaInitial);mem(reference->rhoOmegaInitial);
mem(s->k);mem(reference->k);
mem(s->omega);mem(reference->omega);
mem(s->sstWallDistance);mem(reference->sstWallDistance);
mem(s->sstF1);mem(reference->sstF1);
mem(s->sstF2);mem(reference->sstF2);
mem(s->gradKX);mem(reference->gradKX);
mem(s->gradKY);mem(reference->gradKY);
mem(s->gradKZ);mem(reference->gradKZ);
mem(s->gradOmegaX);mem(reference->gradOmegaX);
mem(s->gradOmegaY);mem(reference->gradOmegaY);
mem(s->gradOmegaZ);mem(reference->gradOmegaZ);
mem(s->sstPhiRhoK);mem(reference->sstPhiRhoK);
mem(s->sstPhiRhoOmega);mem(reference->sstPhiRhoOmega);
mem(s->sstBoundaryKMode);mem(reference->sstBoundaryKMode);
mem(s->sstBoundaryOmegaMode);mem(reference->sstBoundaryOmegaMode);
mem(s->sstBoundaryK);mem(reference->sstBoundaryK);
mem(s->sstBoundaryOmega);mem(reference->sstBoundaryOmega);
mem(s->sstSourceNumber);mem(reference->sstSourceNumber);
mem(s->Ux);mem(reference->Ux);
mem(s->Uy);mem(reference->Uy);
mem(s->Uz);mem(reference->Uz);
mem(s->p);mem(reference->p);
mem(s->Tgas);mem(reference->Tgas);
mem(s->couplingRhoOld);mem(reference->couplingRhoOld);
mem(s->couplingUxOld);mem(reference->couplingUxOld);
mem(s->couplingUyOld);mem(reference->couplingUyOld);
mem(s->couplingUzOld);mem(reference->couplingUzOld);
mem(s->couplingTgasOld);mem(reference->couplingTgasOld);
mem(s->particleHeatCoefficient);mem(reference->particleHeatCoefficient);
mem(s->particleHeatBudget);mem(reference->particleHeatBudget);
mem(s->epsS);mem(reference->epsS);
mem(s->rhoUsx);mem(reference->rhoUsx);
mem(s->rhoUsy);mem(reference->rhoUsy);
mem(s->rhoUsz);mem(reference->rhoUsz);
mem(s->rhoEs);mem(reference->rhoEs);
mem(s->rhoDs);mem(reference->rhoDs);
mem(s->rhoHp);mem(reference->rhoHp);
mem(s->Usx);mem(reference->Usx);
mem(s->Usy);mem(reference->Usy);
mem(s->Usz);mem(reference->Usz);
mem(s->theta);mem(reference->theta);
mem(s->Tp);mem(reference->Tp);
mem(s->dMeanCell);mem(reference->dMeanCell);
mem(s->epsGPrev);mem(reference->epsGPrev);
mem(s->collisionalPressure);mem(reference->collisionalPressure);
mem(s->pressureKickScale);mem(reference->pressureKickScale);
mem(s->pressureDeltaMomX);mem(reference->pressureDeltaMomX);
mem(s->pressureDeltaMomY);mem(reference->pressureDeltaMomY);
mem(s->pressureDeltaMomZ);mem(reference->pressureDeltaMomZ);
mem(s->pressureDeltaEnergy);mem(reference->pressureDeltaEnergy);
mem(s->pressureParticleMoments);mem(reference->pressureParticleMoments);
mem(s->pressureParticleCount);mem(reference->pressureParticleCount);
mem(s->solidPressurePhiMomX);mem(reference->solidPressurePhiMomX);
mem(s->solidPressurePhiMomY);mem(reference->solidPressurePhiMomY);
mem(s->solidPressurePhiMomZ);mem(reference->solidPressurePhiMomZ);
mem(s->solidPressurePhiEnergy);mem(reference->solidPressurePhiEnergy);
mem(s->mobilePackingRho);mem(reference->mobilePackingRho);
mem(s->packingStuckRho);mem(reference->packingStuckRho);
mem(s->mobilePackingMomX);mem(reference->mobilePackingMomX);
mem(s->mobilePackingMomY);mem(reference->mobilePackingMomY);
mem(s->mobilePackingMomZ);mem(reference->mobilePackingMomZ);
mem(s->mobilePackingActiveCellMask);mem(reference->mobilePackingActiveCellMask);
mem(s->mobilePackingCorrectionCellMask);mem(reference->mobilePackingCorrectionCellMask);
mem(s->mobilePackingActiveCellList);mem(reference->mobilePackingActiveCellList);
mem(s->mobilePackingCorrectionCellList);mem(reference->mobilePackingCorrectionCellList);
mem(s->mobilePackingFrontierCurrent);mem(reference->mobilePackingFrontierCurrent);
mem(s->mobilePackingFrontierNext);mem(reference->mobilePackingFrontierNext);
mem(s->mobilePackingActiveCellCount);mem(reference->mobilePackingActiveCellCount);
mem(s->mobilePackingCorrectionCellCount);mem(reference->mobilePackingCorrectionCellCount);
mem(s->mobilePackingFrontierCurrentCount);mem(reference->mobilePackingFrontierCurrentCount);
mem(s->mobilePackingFrontierNextCount);mem(reference->mobilePackingFrontierNextCount);
mem(s->thetaDragAlpha);mem(reference->thetaDragAlpha);
mem(s->momRhoP);mem(reference->momRhoP);
mem(s->momRhoUPx);mem(reference->momRhoUPx);
mem(s->momRhoUPy);mem(reference->momRhoUPy);
mem(s->momRhoUPz);mem(reference->momRhoUPz);
mem(s->momRhoEP);mem(reference->momRhoEP);
mem(s->momRhoPD);mem(reference->momRhoPD);
mem(s->momRhoHpP);mem(reference->momRhoHpP);
mem(s->radiationMobileMass);mem(reference->radiationMobileMass);
mem(s->radiationMobileTemperatureMass);mem(reference->radiationMobileTemperatureMass);
mem(s->radiationMobileDiameterMass);mem(reference->radiationMobileDiameterMass);
mem(s->poolThermalCount);mem(reference->poolThermalCount);
mem(s->poolThermalSumUx);mem(reference->poolThermalSumUx);
mem(s->poolThermalSumUy);mem(reference->poolThermalSumUy);
mem(s->poolThermalSumUz);mem(reference->poolThermalSumUz);
mem(s->poolThermalSumU2);mem(reference->poolThermalSumU2);
mem(s->poissonPoolSampleTargetCount);mem(reference->poissonPoolSampleTargetCount);
mem(s->poissonPoolMass);mem(reference->poissonPoolMass);
mem(s->poissonPoolMomX);mem(reference->poissonPoolMomX);
mem(s->poissonPoolMomY);mem(reference->poissonPoolMomY);
mem(s->poissonPoolMomZ);mem(reference->poissonPoolMomZ);
mem(s->poissonPoolEnergy);mem(reference->poissonPoolEnergy);
mem(s->poissonPoolDiameter);mem(reference->poissonPoolDiameter);
mem(s->poissonPoolDiameter2);mem(reference->poissonPoolDiameter2);
mem(s->px);mem(reference->px);
mem(s->py);mem(reference->py);
mem(s->pz);mem(reference->pz);
mem(s->pux);mem(reference->pux);
mem(s->puy);mem(reference->puy);
mem(s->puz);mem(reference->puz);
mem(s->puxOld);mem(reference->puxOld);
mem(s->puyOld);mem(reference->puyOld);
mem(s->puzOld);mem(reference->puzOld);
mem(s->pT);mem(reference->pT);
mem(s->pThermalEnthalpyResidual);mem(reference->pThermalEnthalpyResidual);
mem(s->pTheta);mem(reference->pTheta);
mem(s->pContactAge);mem(reference->pContactAge);
mem(s->pd);mem(reference->pd);
mem(s->pm);mem(reference->pm);
mem(s->pCellId);mem(reference->pCellId);
mem(s->pStatus);mem(reference->pStatus);
mem(s->pStuck);mem(reference->pStuck);
mem(s->pStuckFaceId);mem(reference->pStuckFaceId);
mem(s->pDepositionArea);mem(reference->pDepositionArea);
mem(s->pContactDuration);mem(reference->pContactDuration);
mem(s->pContactMaximumArea);mem(reference->pContactMaximumArea);
mem(s->pContactPeakFraction);mem(reference->pContactPeakFraction);
mem(s->pColdNodeSpecificEnthalpy);mem(reference->pColdNodeSpecificEnthalpy);
mem(s->pColdRingSolidMass);mem(reference->pColdRingSolidMass);
mem(s->pColdFrozenArea);mem(reference->pColdFrozenArea);
mem(s->pColdContactAge);mem(reference->pColdContactAge);
mem(s->pCold2DNodeSpecificEnthalpy);mem(reference->pCold2DNodeSpecificEnthalpy);
mem(s->pCold2DRingContactAge);mem(reference->pCold2DRingContactAge);
mem(s->pCold2DFrozenArea);mem(reference->pCold2DFrozenArea);
mem(s->pRng);mem(reference->pRng);
mem(s->pOrigId);mem(reference->pOrigId);
mem(s->wallBoundParticleIndex);mem(reference->wallBoundParticleIndex);
mem(s->wallBoundParticleCountDevice);mem(reference->wallBoundParticleCountDevice);
mem(s->sortedParticleIndex);mem(reference->sortedParticleIndex);
mem(s->cellParticleCount);mem(reference->cellParticleCount);
mem(s->cellParticleOffset);mem(reference->cellParticleOffset);
mem(s->compactCellOffset);mem(reference->compactCellOffset);
mem(s->cellParticleWrite);mem(reference->cellParticleWrite);
mem(s->preBaseCellOffset);mem(reference->preBaseCellOffset);
mem(s->preBaseParticleCountDevice);mem(reference->preBaseParticleCountDevice);
mem(s->cellScanTempStorage);mem(reference->cellScanTempStorage);
mem(s->compactCountDevice);mem(reference->compactCountDevice);
mem(s->compactSelectTempStorage);mem(reference->compactSelectTempStorage);
mem(s->csrCellTaskCount);mem(reference->csrCellTaskCount);
mem(s->csrCellTaskOffset);mem(reference->csrCellTaskOffset);
mem(s->csrMultiTaskCellList);mem(reference->csrMultiTaskCellList);
mem(s->csrHeavyCellList);mem(reference->csrHeavyCellList);
mem(s->csrHeavyTaskCell);mem(reference->csrHeavyTaskCell);
mem(s->csrHeavyTaskBegin);mem(reference->csrHeavyTaskBegin);
mem(s->csrHeavyTaskEnd);mem(reference->csrHeavyTaskEnd);
mem(s->csrHeavyCellTaskStart);mem(reference->csrHeavyCellTaskStart);
mem(s->csrHeavyCellTaskCount);mem(reference->csrHeavyCellTaskCount);
mem(s->csrHeavyPartials);mem(reference->csrHeavyPartials);
mem(s->csrHeavyInjectionTaskCount);mem(reference->csrHeavyInjectionTaskCount);
mem(s->csrHeavyInjectionTaskCursor);mem(reference->csrHeavyInjectionTaskCursor);
mem(s->csrHeavyInjectionTaskCell);mem(reference->csrHeavyInjectionTaskCell);
mem(s->csrHeavyInjectionTaskBegin);mem(reference->csrHeavyInjectionTaskBegin);
mem(s->csrHeavyInjectionTaskEnd);mem(reference->csrHeavyInjectionTaskEnd);
mem(s->csrHeavyInjectionCellTaskStart);mem(reference->csrHeavyInjectionCellTaskStart);
mem(s->csrHeavyInjectionCellTaskCount);mem(reference->csrHeavyInjectionCellTaskCount);
mem(s->csrHeavyInjectionPartials);mem(reference->csrHeavyInjectionPartials);
mem(s->sourceCell);mem(reference->sourceCell);
mem(s->sourceFace);mem(reference->sourceFace);
mem(s->sourcePx);mem(reference->sourcePx);
mem(s->sourcePy);mem(reference->sourcePy);
mem(s->sourcePz);mem(reference->sourcePz);
mem(s->sourceUx);mem(reference->sourceUx);
mem(s->sourceUy);mem(reference->sourceUy);
mem(s->sourceUz);mem(reference->sourceUz);
mem(s->sourceT);mem(reference->sourceT);
mem(s->sourceTheta);mem(reference->sourceTheta);
mem(s->sourceD);mem(reference->sourceD);
mem(s->sourceMassRate);mem(reference->sourceMassRate);
mem(s->sourceResidualMass);mem(reference->sourceResidualMass);
mem(s->compactPx);mem(reference->compactPx);
mem(s->compactPy);mem(reference->compactPy);
mem(s->compactPz);mem(reference->compactPz);
mem(s->compactPux);mem(reference->compactPux);
mem(s->compactPuxOld);mem(reference->compactPuxOld);
mem(s->compactPuy);mem(reference->compactPuy);
mem(s->compactPuyOld);mem(reference->compactPuyOld);
mem(s->compactPuz);mem(reference->compactPuz);
mem(s->compactPuzOld);mem(reference->compactPuzOld);
mem(s->compactPT);mem(reference->compactPT);
mem(s->compactPThermalEnthalpyResidual);mem(reference->compactPThermalEnthalpyResidual);
mem(s->compactPTheta);mem(reference->compactPTheta);
mem(s->compactPContactAge);mem(reference->compactPContactAge);
mem(s->compactPd);mem(reference->compactPd);
mem(s->compactPm);mem(reference->compactPm);
mem(s->compactPCellId);mem(reference->compactPCellId);
mem(s->compactPStatus);mem(reference->compactPStatus);
mem(s->compactPStuck);mem(reference->compactPStuck);
mem(s->compactPStuckFaceId);mem(reference->compactPStuckFaceId);
mem(s->compactPDepositionArea);mem(reference->compactPDepositionArea);
mem(s->compactPContactDuration);mem(reference->compactPContactDuration);
mem(s->compactPContactMaximumArea);mem(reference->compactPContactMaximumArea);
mem(s->compactPContactPeakFraction);mem(reference->compactPContactPeakFraction);
mem(s->compactPColdNodeSpecificEnthalpy);mem(reference->compactPColdNodeSpecificEnthalpy);
mem(s->compactPColdRingSolidMass);mem(reference->compactPColdRingSolidMass);
mem(s->compactPColdFrozenArea);mem(reference->compactPColdFrozenArea);
mem(s->compactPColdContactAge);mem(reference->compactPColdContactAge);
mem(s->compactPCold2DNodeSpecificEnthalpy);mem(reference->compactPCold2DNodeSpecificEnthalpy);
mem(s->compactPCold2DRingContactAge);mem(reference->compactPCold2DRingContactAge);
mem(s->compactPCold2DFrozenArea);mem(reference->compactPCold2DFrozenArea);
mem(s->compactPRng);mem(reference->compactPRng);
mem(s->compactPOrigId);mem(reference->compactPOrigId);
s->nCells=16;s->nFaces=40;s->nInternalFaces=8;s->rhoSolid=3000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->rhoMin=1e-12;s->Rgas=1;s->TgasMin=.001;s->particleDiameterFallback=1e-4;s->collisionalPressureEnabled=1;s->collisionalRestitution=.9;
for(int c=0;c<16;c++){
 s->rho[c]=1+.125*c;s->Ux[c]=2+.25*c;s->Uy[c]=-.5;s->Uz[c]=.25;s->p[c]=2+.25*c;s->Tgas[c]=300+c;
 s->momRhoP[c]=(c==0?0:100+.125*c);s->momRhoUPx[c]=.25*s->momRhoP[c];s->momRhoUPy[c]=-.125*s->momRhoP[c];s->momRhoUPz[c]=0;s->momRhoEP[c]=(c==2?1e-30:5)*s->momRhoP[c];s->momRhoPD[c]=1e-4*s->momRhoP[c];
 s->V[c]=1+.0625*c;s->cellPlaneStart[c]=3*c;s->cellPlaneCount[c]=3;s->cellFaceId[3*c]=c%8;s->cellFaceId[3*c+1]=-1;s->cellFaceId[3*c+2]=(c+2)%8;
}
s->momRhoP[3]=NAN;s->momRhoPD[4]=NAN;s->momRhoEP[5]=INFINITY;
for(int f=0;f<40;f++){
 s->faceOwner[f]=f%16;s->faceNeighbour[f]=(f+1)%16;s->facePeriodicPair[f]=f>=8&&f<16?(f^1):-1;s->faceWeight[f]=.25*(f%5);s->riemannBoundaryKind[f]=f%5;s->riemannBoundaryUFix[f]=f%2;s->riemannBoundaryTFix[f]=f%2;
 s->magSf[f]=1;s->Sfx[f]=1;s->riemannBoundaryUx[f]=.5;s->riemannBoundaryT[f]=500;s->solidPressurePhiMomX[f]=.125*(f-4);s->solidPressurePhiEnergy[f]=.25*(f-2);
}
double*values; if(cudaMallocManaged(&values,512*32*sizeof(double))!=cudaSuccess)abort();
primitives<<<4,128>>>(s,values);deviceSync();for(int i=0;i<512;i++)for(int j=0;j<9;j++)equal(values[32*i+2*j],values[32*i+2*j+1],"moment/face primitive");
puts("PASS bitwise moments and face states: invalid cells, nonfinite inputs, internal faces from both sides, periodic owners and wall kinds");
dragRates<<<4,128>>>(values);deviceSync();for(int i=0;i<512*4;i++)equal(values[2*i],values[2*i+1],"drag policy rate");
puts("PASS 2048 bitwise drag rates including zero slip, Re/alpha thresholds and denominator bounds");
// Pressure publication uses a finite positive fixture; primitive tests above
// separately cover nonfinite moment sanitization.
for(int c=0;c<16;c++){s->momRhoP[c]=100;s->momRhoEP[c]=500;s->momRhoUPx[c]=25;s->momRhoUPy[c]=-12.5;s->momRhoUPz[c]=0;}
{DeviceState copy=*s;
copy.csrMaximumOccupancy=reference->csrMaximumOccupancy;std::memcpy(copy.csrMaximumOccupancy,s->csrMaximumOccupancy,1024*sizeof(*s->csrMaximumOccupancy));
copy.csrHeavyTaskCount=reference->csrHeavyTaskCount;std::memcpy(copy.csrHeavyTaskCount,s->csrHeavyTaskCount,1024*sizeof(*s->csrHeavyTaskCount));
copy.csrHeavyCellCount=reference->csrHeavyCellCount;std::memcpy(copy.csrHeavyCellCount,s->csrHeavyCellCount,1024*sizeof(*s->csrHeavyCellCount));
copy.csrHeavyTaskCursor=reference->csrHeavyTaskCursor;std::memcpy(copy.csrHeavyTaskCursor,s->csrHeavyTaskCursor,1024*sizeof(*s->csrHeavyTaskCursor));
copy.particleCountDevice=reference->particleCountDevice;std::memcpy(copy.particleCountDevice,s->particleCountDevice,1024*sizeof(*s->particleCountDevice));
copy.faceOwner=reference->faceOwner;std::memcpy(copy.faceOwner,s->faceOwner,1024*sizeof(*s->faceOwner));
copy.faceNeighbour=reference->faceNeighbour;std::memcpy(copy.faceNeighbour,s->faceNeighbour,1024*sizeof(*s->faceNeighbour));
copy.facePeriodicPair=reference->facePeriodicPair;std::memcpy(copy.facePeriodicPair,s->facePeriodicPair,1024*sizeof(*s->facePeriodicPair));
copy.facePeriodicDx=reference->facePeriodicDx;std::memcpy(copy.facePeriodicDx,s->facePeriodicDx,1024*sizeof(*s->facePeriodicDx));
copy.facePeriodicDy=reference->facePeriodicDy;std::memcpy(copy.facePeriodicDy,s->facePeriodicDy,1024*sizeof(*s->facePeriodicDy));
copy.facePeriodicDz=reference->facePeriodicDz;std::memcpy(copy.facePeriodicDz,s->facePeriodicDz,1024*sizeof(*s->facePeriodicDz));
copy.V=reference->V;std::memcpy(copy.V,s->V,1024*sizeof(*s->V));
copy.Cx=reference->Cx;std::memcpy(copy.Cx,s->Cx,1024*sizeof(*s->Cx));
copy.Cy=reference->Cy;std::memcpy(copy.Cy,s->Cy,1024*sizeof(*s->Cy));
copy.Cz=reference->Cz;std::memcpy(copy.Cz,s->Cz,1024*sizeof(*s->Cz));
copy.faceCx=reference->faceCx;std::memcpy(copy.faceCx,s->faceCx,1024*sizeof(*s->faceCx));
copy.faceCy=reference->faceCy;std::memcpy(copy.faceCy,s->faceCy,1024*sizeof(*s->faceCy));
copy.faceCz=reference->faceCz;std::memcpy(copy.faceCz,s->faceCz,1024*sizeof(*s->faceCz));
copy.Sfx=reference->Sfx;std::memcpy(copy.Sfx,s->Sfx,1024*sizeof(*s->Sfx));
copy.Sfy=reference->Sfy;std::memcpy(copy.Sfy,s->Sfy,1024*sizeof(*s->Sfy));
copy.Sfz=reference->Sfz;std::memcpy(copy.Sfz,s->Sfz,1024*sizeof(*s->Sfz));
copy.magSf=reference->magSf;std::memcpy(copy.magSf,s->magSf,1024*sizeof(*s->magSf));
copy.deltaCoeffs=reference->deltaCoeffs;std::memcpy(copy.deltaCoeffs,s->deltaCoeffs,1024*sizeof(*s->deltaCoeffs));
copy.faceWeight=reference->faceWeight;std::memcpy(copy.faceWeight,s->faceWeight,1024*sizeof(*s->faceWeight));
copy.cellLength=reference->cellLength;std::memcpy(copy.cellLength,s->cellLength,1024*sizeof(*s->cellLength));
copy.cellPlaneStart=reference->cellPlaneStart;std::memcpy(copy.cellPlaneStart,s->cellPlaneStart,1024*sizeof(*s->cellPlaneStart));
copy.cellPlaneCount=reference->cellPlaneCount;std::memcpy(copy.cellPlaneCount,s->cellPlaneCount,1024*sizeof(*s->cellPlaneCount));
copy.cellFaceId=reference->cellFaceId;std::memcpy(copy.cellFaceId,s->cellFaceId,1024*sizeof(*s->cellFaceId));
copy.cellFaceNeighbor=reference->cellFaceNeighbor;std::memcpy(copy.cellFaceNeighbor,s->cellFaceNeighbor,1024*sizeof(*s->cellFaceNeighbor));
copy.cellFaceKind=reference->cellFaceKind;std::memcpy(copy.cellFaceKind,s->cellFaceKind,1024*sizeof(*s->cellFaceKind));
copy.cellFaceRestitution=reference->cellFaceRestitution;std::memcpy(copy.cellFaceRestitution,s->cellFaceRestitution,1024*sizeof(*s->cellFaceRestitution));
copy.cellFaceTangential=reference->cellFaceTangential;std::memcpy(copy.cellFaceTangential,s->cellFaceTangential,1024*sizeof(*s->cellFaceTangential));
copy.planeNx=reference->planeNx;std::memcpy(copy.planeNx,s->planeNx,1024*sizeof(*s->planeNx));
copy.planeNy=reference->planeNy;std::memcpy(copy.planeNy,s->planeNy,1024*sizeof(*s->planeNy));
copy.planeNz=reference->planeNz;std::memcpy(copy.planeNz,s->planeNz,1024*sizeof(*s->planeNz));
copy.planeD=reference->planeD;std::memcpy(copy.planeD,s->planeD,1024*sizeof(*s->planeD));
copy.rho=reference->rho;std::memcpy(copy.rho,s->rho,1024*sizeof(*s->rho));
copy.rhoUx=reference->rhoUx;std::memcpy(copy.rhoUx,s->rhoUx,1024*sizeof(*s->rhoUx));
copy.rhoUy=reference->rhoUy;std::memcpy(copy.rhoUy,s->rhoUy,1024*sizeof(*s->rhoUy));
copy.rhoUz=reference->rhoUz;std::memcpy(copy.rhoUz,s->rhoUz,1024*sizeof(*s->rhoUz));
copy.rhoE=reference->rhoE;std::memcpy(copy.rhoE,s->rhoE,1024*sizeof(*s->rhoE));
copy.rhoNext=reference->rhoNext;std::memcpy(copy.rhoNext,s->rhoNext,1024*sizeof(*s->rhoNext));
copy.rhoUxNext=reference->rhoUxNext;std::memcpy(copy.rhoUxNext,s->rhoUxNext,1024*sizeof(*s->rhoUxNext));
copy.rhoUyNext=reference->rhoUyNext;std::memcpy(copy.rhoUyNext,s->rhoUyNext,1024*sizeof(*s->rhoUyNext));
copy.rhoUzNext=reference->rhoUzNext;std::memcpy(copy.rhoUzNext,s->rhoUzNext,1024*sizeof(*s->rhoUzNext));
copy.rhoENext=reference->rhoENext;std::memcpy(copy.rhoENext,s->rhoENext,1024*sizeof(*s->rhoENext));
copy.gasPhiRho=reference->gasPhiRho;std::memcpy(copy.gasPhiRho,s->gasPhiRho,1024*sizeof(*s->gasPhiRho));
copy.gasPhiRhoUx=reference->gasPhiRhoUx;std::memcpy(copy.gasPhiRhoUx,s->gasPhiRhoUx,1024*sizeof(*s->gasPhiRhoUx));
copy.gasPhiRhoUy=reference->gasPhiRhoUy;std::memcpy(copy.gasPhiRhoUy,s->gasPhiRhoUy,1024*sizeof(*s->gasPhiRhoUy));
copy.gasPhiRhoUz=reference->gasPhiRhoUz;std::memcpy(copy.gasPhiRhoUz,s->gasPhiRhoUz,1024*sizeof(*s->gasPhiRhoUz));
copy.gasPhiRhoE=reference->gasPhiRhoE;std::memcpy(copy.gasPhiRhoE,s->gasPhiRhoE,1024*sizeof(*s->gasPhiRhoE));
copy.gasFluxPositivityScale=reference->gasFluxPositivityScale;std::memcpy(copy.gasFluxPositivityScale,s->gasFluxPositivityScale,1024*sizeof(*s->gasFluxPositivityScale));
copy.gasHllcAdcSensor=reference->gasHllcAdcSensor;std::memcpy(copy.gasHllcAdcSensor,s->gasHllcAdcSensor,1024*sizeof(*s->gasHllcAdcSensor));
copy.particleStuckCandidateMask=reference->particleStuckCandidateMask;std::memcpy(copy.particleStuckCandidateMask,s->particleStuckCandidateMask,1024*sizeof(*s->particleStuckCandidateMask));
copy.particleWallDepositedEnergy=reference->particleWallDepositedEnergy;std::memcpy(copy.particleWallDepositedEnergy,s->particleWallDepositedEnergy,1024*sizeof(*s->particleWallDepositedEnergy));
copy.particleWallReflectedEnergy=reference->particleWallReflectedEnergy;std::memcpy(copy.particleWallReflectedEnergy,s->particleWallReflectedEnergy,1024*sizeof(*s->particleWallReflectedEnergy));
copy.particleWallRepresentedContactArea=reference->particleWallRepresentedContactArea;std::memcpy(copy.particleWallRepresentedContactArea,s->particleWallRepresentedContactArea,1024*sizeof(*s->particleWallRepresentedContactArea));
copy.particleWallContactAreaScale=reference->particleWallContactAreaScale;std::memcpy(copy.particleWallContactAreaScale,s->particleWallContactAreaScale,1024*sizeof(*s->particleWallContactAreaScale));
copy.particleWallEffusivityByFace=reference->particleWallEffusivityByFace;std::memcpy(copy.particleWallEffusivityByFace,s->particleWallEffusivityByFace,1024*sizeof(*s->particleWallEffusivityByFace));
copy.finiteContactRateTable=reference->finiteContactRateTable;std::memcpy(copy.finiteContactRateTable,s->finiteContactRateTable,1024*sizeof(*s->finiteContactRateTable));
copy.gasBoundaryKind=reference->gasBoundaryKind;std::memcpy(copy.gasBoundaryKind,s->gasBoundaryKind,1024*sizeof(*s->gasBoundaryKind));
copy.gasBoundaryRhoFix=reference->gasBoundaryRhoFix;std::memcpy(copy.gasBoundaryRhoFix,s->gasBoundaryRhoFix,1024*sizeof(*s->gasBoundaryRhoFix));
copy.gasBoundaryUFix=reference->gasBoundaryUFix;std::memcpy(copy.gasBoundaryUFix,s->gasBoundaryUFix,1024*sizeof(*s->gasBoundaryUFix));
copy.gasBoundaryPFix=reference->gasBoundaryPFix;std::memcpy(copy.gasBoundaryPFix,s->gasBoundaryPFix,1024*sizeof(*s->gasBoundaryPFix));
copy.gasBoundaryTFix=reference->gasBoundaryTFix;std::memcpy(copy.gasBoundaryTFix,s->gasBoundaryTFix,1024*sizeof(*s->gasBoundaryTFix));
copy.gasBoundaryPWave=reference->gasBoundaryPWave;std::memcpy(copy.gasBoundaryPWave,s->gasBoundaryPWave,1024*sizeof(*s->gasBoundaryPWave));
copy.gasBoundaryPWaveGamma=reference->gasBoundaryPWaveGamma;std::memcpy(copy.gasBoundaryPWaveGamma,s->gasBoundaryPWaveGamma,1024*sizeof(*s->gasBoundaryPWaveGamma));
copy.gasBoundaryPWaveFieldInf=reference->gasBoundaryPWaveFieldInf;std::memcpy(copy.gasBoundaryPWaveFieldInf,s->gasBoundaryPWaveFieldInf,1024*sizeof(*s->gasBoundaryPWaveFieldInf));
copy.gasBoundaryPWaveLInf=reference->gasBoundaryPWaveLInf;std::memcpy(copy.gasBoundaryPWaveLInf,s->gasBoundaryPWaveLInf,1024*sizeof(*s->gasBoundaryPWaveLInf));
copy.gasBoundaryRho=reference->gasBoundaryRho;std::memcpy(copy.gasBoundaryRho,s->gasBoundaryRho,1024*sizeof(*s->gasBoundaryRho));
copy.gasBoundaryUx=reference->gasBoundaryUx;std::memcpy(copy.gasBoundaryUx,s->gasBoundaryUx,1024*sizeof(*s->gasBoundaryUx));
copy.gasBoundaryUy=reference->gasBoundaryUy;std::memcpy(copy.gasBoundaryUy,s->gasBoundaryUy,1024*sizeof(*s->gasBoundaryUy));
copy.gasBoundaryUz=reference->gasBoundaryUz;std::memcpy(copy.gasBoundaryUz,s->gasBoundaryUz,1024*sizeof(*s->gasBoundaryUz));
copy.gasBoundaryP=reference->gasBoundaryP;std::memcpy(copy.gasBoundaryP,s->gasBoundaryP,1024*sizeof(*s->gasBoundaryP));
copy.gasBoundaryT=reference->gasBoundaryT;std::memcpy(copy.gasBoundaryT,s->gasBoundaryT,1024*sizeof(*s->gasBoundaryT));
copy.riemannBoundaryKind=reference->riemannBoundaryKind;std::memcpy(copy.riemannBoundaryKind,s->riemannBoundaryKind,1024*sizeof(*s->riemannBoundaryKind));
copy.riemannBoundaryRhoFix=reference->riemannBoundaryRhoFix;std::memcpy(copy.riemannBoundaryRhoFix,s->riemannBoundaryRhoFix,1024*sizeof(*s->riemannBoundaryRhoFix));
copy.riemannBoundaryUFix=reference->riemannBoundaryUFix;std::memcpy(copy.riemannBoundaryUFix,s->riemannBoundaryUFix,1024*sizeof(*s->riemannBoundaryUFix));
copy.riemannBoundaryPFix=reference->riemannBoundaryPFix;std::memcpy(copy.riemannBoundaryPFix,s->riemannBoundaryPFix,1024*sizeof(*s->riemannBoundaryPFix));
copy.riemannBoundaryTFix=reference->riemannBoundaryTFix;std::memcpy(copy.riemannBoundaryTFix,s->riemannBoundaryTFix,1024*sizeof(*s->riemannBoundaryTFix));
copy.riemannBoundaryPWave=reference->riemannBoundaryPWave;std::memcpy(copy.riemannBoundaryPWave,s->riemannBoundaryPWave,1024*sizeof(*s->riemannBoundaryPWave));
copy.riemannBoundaryPWaveGamma=reference->riemannBoundaryPWaveGamma;std::memcpy(copy.riemannBoundaryPWaveGamma,s->riemannBoundaryPWaveGamma,1024*sizeof(*s->riemannBoundaryPWaveGamma));
copy.riemannBoundaryPWaveFieldInf=reference->riemannBoundaryPWaveFieldInf;std::memcpy(copy.riemannBoundaryPWaveFieldInf,s->riemannBoundaryPWaveFieldInf,1024*sizeof(*s->riemannBoundaryPWaveFieldInf));
copy.riemannBoundaryPWaveLInf=reference->riemannBoundaryPWaveLInf;std::memcpy(copy.riemannBoundaryPWaveLInf,s->riemannBoundaryPWaveLInf,1024*sizeof(*s->riemannBoundaryPWaveLInf));
copy.riemannBoundaryRho=reference->riemannBoundaryRho;std::memcpy(copy.riemannBoundaryRho,s->riemannBoundaryRho,1024*sizeof(*s->riemannBoundaryRho));
copy.riemannBoundaryUx=reference->riemannBoundaryUx;std::memcpy(copy.riemannBoundaryUx,s->riemannBoundaryUx,1024*sizeof(*s->riemannBoundaryUx));
copy.riemannBoundaryUy=reference->riemannBoundaryUy;std::memcpy(copy.riemannBoundaryUy,s->riemannBoundaryUy,1024*sizeof(*s->riemannBoundaryUy));
copy.riemannBoundaryUz=reference->riemannBoundaryUz;std::memcpy(copy.riemannBoundaryUz,s->riemannBoundaryUz,1024*sizeof(*s->riemannBoundaryUz));
copy.riemannBoundaryP=reference->riemannBoundaryP;std::memcpy(copy.riemannBoundaryP,s->riemannBoundaryP,1024*sizeof(*s->riemannBoundaryP));
copy.riemannBoundaryT=reference->riemannBoundaryT;std::memcpy(copy.riemannBoundaryT,s->riemannBoundaryT,1024*sizeof(*s->riemannBoundaryT));
copy.scheduledInletFaceMask=reference->scheduledInletFaceMask;std::memcpy(copy.scheduledInletFaceMask,s->scheduledInletFaceMask,1024*sizeof(*s->scheduledInletFaceMask));
copy.pressureScheduleTimes=reference->pressureScheduleTimes;std::memcpy(copy.pressureScheduleTimes,s->pressureScheduleTimes,1024*sizeof(*s->pressureScheduleTimes));
copy.pressureScheduleValues=reference->pressureScheduleValues;std::memcpy(copy.pressureScheduleValues,s->pressureScheduleValues,1024*sizeof(*s->pressureScheduleValues));
copy.volumeFractionScheduleTimes=reference->volumeFractionScheduleTimes;std::memcpy(copy.volumeFractionScheduleTimes,s->volumeFractionScheduleTimes,1024*sizeof(*s->volumeFractionScheduleTimes));
copy.volumeFractionScheduleValues=reference->volumeFractionScheduleValues;std::memcpy(copy.volumeFractionScheduleValues,s->volumeFractionScheduleValues,1024*sizeof(*s->volumeFractionScheduleValues));
copy.gradPx=reference->gradPx;std::memcpy(copy.gradPx,s->gradPx,1024*sizeof(*s->gradPx));
copy.gradPy=reference->gradPy;std::memcpy(copy.gradPy,s->gradPy,1024*sizeof(*s->gradPy));
copy.gradPz=reference->gradPz;std::memcpy(copy.gradPz,s->gradPz,1024*sizeof(*s->gradPz));
copy.gradRhoX=reference->gradRhoX;std::memcpy(copy.gradRhoX,s->gradRhoX,1024*sizeof(*s->gradRhoX));
copy.gradRhoY=reference->gradRhoY;std::memcpy(copy.gradRhoY,s->gradRhoY,1024*sizeof(*s->gradRhoY));
copy.gradRhoZ=reference->gradRhoZ;std::memcpy(copy.gradRhoZ,s->gradRhoZ,1024*sizeof(*s->gradRhoZ));
copy.gradUxX=reference->gradUxX;std::memcpy(copy.gradUxX,s->gradUxX,1024*sizeof(*s->gradUxX));
copy.gradUxY=reference->gradUxY;std::memcpy(copy.gradUxY,s->gradUxY,1024*sizeof(*s->gradUxY));
copy.gradUxZ=reference->gradUxZ;std::memcpy(copy.gradUxZ,s->gradUxZ,1024*sizeof(*s->gradUxZ));
copy.gradUyX=reference->gradUyX;std::memcpy(copy.gradUyX,s->gradUyX,1024*sizeof(*s->gradUyX));
copy.gradUyY=reference->gradUyY;std::memcpy(copy.gradUyY,s->gradUyY,1024*sizeof(*s->gradUyY));
copy.gradUyZ=reference->gradUyZ;std::memcpy(copy.gradUyZ,s->gradUyZ,1024*sizeof(*s->gradUyZ));
copy.gradUzX=reference->gradUzX;std::memcpy(copy.gradUzX,s->gradUzX,1024*sizeof(*s->gradUzX));
copy.gradUzY=reference->gradUzY;std::memcpy(copy.gradUzY,s->gradUzY,1024*sizeof(*s->gradUzY));
copy.gradUzZ=reference->gradUzZ;std::memcpy(copy.gradUzZ,s->gradUzZ,1024*sizeof(*s->gradUzZ));
copy.gradTX=reference->gradTX;std::memcpy(copy.gradTX,s->gradTX,1024*sizeof(*s->gradTX));
copy.gradTY=reference->gradTY;std::memcpy(copy.gradTY,s->gradTY,1024*sizeof(*s->gradTY));
copy.gradTZ=reference->gradTZ;std::memcpy(copy.gradTZ,s->gradTZ,1024*sizeof(*s->gradTZ));
copy.gasGradientLimiterRho=reference->gasGradientLimiterRho;std::memcpy(copy.gasGradientLimiterRho,s->gasGradientLimiterRho,1024*sizeof(*s->gasGradientLimiterRho));
copy.gasGradientLimiterUx=reference->gasGradientLimiterUx;std::memcpy(copy.gasGradientLimiterUx,s->gasGradientLimiterUx,1024*sizeof(*s->gasGradientLimiterUx));
copy.gasGradientLimiterUy=reference->gasGradientLimiterUy;std::memcpy(copy.gasGradientLimiterUy,s->gasGradientLimiterUy,1024*sizeof(*s->gasGradientLimiterUy));
copy.gasGradientLimiterUz=reference->gasGradientLimiterUz;std::memcpy(copy.gasGradientLimiterUz,s->gasGradientLimiterUz,1024*sizeof(*s->gasGradientLimiterUz));
copy.gasGradientLimiterP=reference->gasGradientLimiterP;std::memcpy(copy.gasGradientLimiterP,s->gasGradientLimiterP,1024*sizeof(*s->gasGradientLimiterP));
copy.gasGradientLimiterT=reference->gasGradientLimiterT;std::memcpy(copy.gasGradientLimiterT,s->gasGradientLimiterT,1024*sizeof(*s->gasGradientLimiterT));
copy.nut=reference->nut;std::memcpy(copy.nut,s->nut,1024*sizeof(*s->nut));
copy.gasDiffusionNumber=reference->gasDiffusionNumber;std::memcpy(copy.gasDiffusionNumber,s->gasDiffusionNumber,1024*sizeof(*s->gasDiffusionNumber));
copy.rhoK=reference->rhoK;std::memcpy(copy.rhoK,s->rhoK,1024*sizeof(*s->rhoK));
copy.rhoOmega=reference->rhoOmega;std::memcpy(copy.rhoOmega,s->rhoOmega,1024*sizeof(*s->rhoOmega));
copy.rhoKInitial=reference->rhoKInitial;std::memcpy(copy.rhoKInitial,s->rhoKInitial,1024*sizeof(*s->rhoKInitial));
copy.rhoOmegaInitial=reference->rhoOmegaInitial;std::memcpy(copy.rhoOmegaInitial,s->rhoOmegaInitial,1024*sizeof(*s->rhoOmegaInitial));
copy.k=reference->k;std::memcpy(copy.k,s->k,1024*sizeof(*s->k));
copy.omega=reference->omega;std::memcpy(copy.omega,s->omega,1024*sizeof(*s->omega));
copy.sstWallDistance=reference->sstWallDistance;std::memcpy(copy.sstWallDistance,s->sstWallDistance,1024*sizeof(*s->sstWallDistance));
copy.sstF1=reference->sstF1;std::memcpy(copy.sstF1,s->sstF1,1024*sizeof(*s->sstF1));
copy.sstF2=reference->sstF2;std::memcpy(copy.sstF2,s->sstF2,1024*sizeof(*s->sstF2));
copy.gradKX=reference->gradKX;std::memcpy(copy.gradKX,s->gradKX,1024*sizeof(*s->gradKX));
copy.gradKY=reference->gradKY;std::memcpy(copy.gradKY,s->gradKY,1024*sizeof(*s->gradKY));
copy.gradKZ=reference->gradKZ;std::memcpy(copy.gradKZ,s->gradKZ,1024*sizeof(*s->gradKZ));
copy.gradOmegaX=reference->gradOmegaX;std::memcpy(copy.gradOmegaX,s->gradOmegaX,1024*sizeof(*s->gradOmegaX));
copy.gradOmegaY=reference->gradOmegaY;std::memcpy(copy.gradOmegaY,s->gradOmegaY,1024*sizeof(*s->gradOmegaY));
copy.gradOmegaZ=reference->gradOmegaZ;std::memcpy(copy.gradOmegaZ,s->gradOmegaZ,1024*sizeof(*s->gradOmegaZ));
copy.sstPhiRhoK=reference->sstPhiRhoK;std::memcpy(copy.sstPhiRhoK,s->sstPhiRhoK,1024*sizeof(*s->sstPhiRhoK));
copy.sstPhiRhoOmega=reference->sstPhiRhoOmega;std::memcpy(copy.sstPhiRhoOmega,s->sstPhiRhoOmega,1024*sizeof(*s->sstPhiRhoOmega));
copy.sstBoundaryKMode=reference->sstBoundaryKMode;std::memcpy(copy.sstBoundaryKMode,s->sstBoundaryKMode,1024*sizeof(*s->sstBoundaryKMode));
copy.sstBoundaryOmegaMode=reference->sstBoundaryOmegaMode;std::memcpy(copy.sstBoundaryOmegaMode,s->sstBoundaryOmegaMode,1024*sizeof(*s->sstBoundaryOmegaMode));
copy.sstBoundaryK=reference->sstBoundaryK;std::memcpy(copy.sstBoundaryK,s->sstBoundaryK,1024*sizeof(*s->sstBoundaryK));
copy.sstBoundaryOmega=reference->sstBoundaryOmega;std::memcpy(copy.sstBoundaryOmega,s->sstBoundaryOmega,1024*sizeof(*s->sstBoundaryOmega));
copy.sstSourceNumber=reference->sstSourceNumber;std::memcpy(copy.sstSourceNumber,s->sstSourceNumber,1024*sizeof(*s->sstSourceNumber));
copy.Ux=reference->Ux;std::memcpy(copy.Ux,s->Ux,1024*sizeof(*s->Ux));
copy.Uy=reference->Uy;std::memcpy(copy.Uy,s->Uy,1024*sizeof(*s->Uy));
copy.Uz=reference->Uz;std::memcpy(copy.Uz,s->Uz,1024*sizeof(*s->Uz));
copy.p=reference->p;std::memcpy(copy.p,s->p,1024*sizeof(*s->p));
copy.Tgas=reference->Tgas;std::memcpy(copy.Tgas,s->Tgas,1024*sizeof(*s->Tgas));
copy.couplingRhoOld=reference->couplingRhoOld;std::memcpy(copy.couplingRhoOld,s->couplingRhoOld,1024*sizeof(*s->couplingRhoOld));
copy.couplingUxOld=reference->couplingUxOld;std::memcpy(copy.couplingUxOld,s->couplingUxOld,1024*sizeof(*s->couplingUxOld));
copy.couplingUyOld=reference->couplingUyOld;std::memcpy(copy.couplingUyOld,s->couplingUyOld,1024*sizeof(*s->couplingUyOld));
copy.couplingUzOld=reference->couplingUzOld;std::memcpy(copy.couplingUzOld,s->couplingUzOld,1024*sizeof(*s->couplingUzOld));
copy.couplingTgasOld=reference->couplingTgasOld;std::memcpy(copy.couplingTgasOld,s->couplingTgasOld,1024*sizeof(*s->couplingTgasOld));
copy.particleHeatCoefficient=reference->particleHeatCoefficient;std::memcpy(copy.particleHeatCoefficient,s->particleHeatCoefficient,1024*sizeof(*s->particleHeatCoefficient));
copy.particleHeatBudget=reference->particleHeatBudget;std::memcpy(copy.particleHeatBudget,s->particleHeatBudget,1024*sizeof(*s->particleHeatBudget));
copy.epsS=reference->epsS;std::memcpy(copy.epsS,s->epsS,1024*sizeof(*s->epsS));
copy.rhoUsx=reference->rhoUsx;std::memcpy(copy.rhoUsx,s->rhoUsx,1024*sizeof(*s->rhoUsx));
copy.rhoUsy=reference->rhoUsy;std::memcpy(copy.rhoUsy,s->rhoUsy,1024*sizeof(*s->rhoUsy));
copy.rhoUsz=reference->rhoUsz;std::memcpy(copy.rhoUsz,s->rhoUsz,1024*sizeof(*s->rhoUsz));
copy.rhoEs=reference->rhoEs;std::memcpy(copy.rhoEs,s->rhoEs,1024*sizeof(*s->rhoEs));
copy.rhoDs=reference->rhoDs;std::memcpy(copy.rhoDs,s->rhoDs,1024*sizeof(*s->rhoDs));
copy.rhoHp=reference->rhoHp;std::memcpy(copy.rhoHp,s->rhoHp,1024*sizeof(*s->rhoHp));
copy.Usx=reference->Usx;std::memcpy(copy.Usx,s->Usx,1024*sizeof(*s->Usx));
copy.Usy=reference->Usy;std::memcpy(copy.Usy,s->Usy,1024*sizeof(*s->Usy));
copy.Usz=reference->Usz;std::memcpy(copy.Usz,s->Usz,1024*sizeof(*s->Usz));
copy.theta=reference->theta;std::memcpy(copy.theta,s->theta,1024*sizeof(*s->theta));
copy.Tp=reference->Tp;std::memcpy(copy.Tp,s->Tp,1024*sizeof(*s->Tp));
copy.dMeanCell=reference->dMeanCell;std::memcpy(copy.dMeanCell,s->dMeanCell,1024*sizeof(*s->dMeanCell));
copy.epsGPrev=reference->epsGPrev;std::memcpy(copy.epsGPrev,s->epsGPrev,1024*sizeof(*s->epsGPrev));
copy.collisionalPressure=reference->collisionalPressure;std::memcpy(copy.collisionalPressure,s->collisionalPressure,1024*sizeof(*s->collisionalPressure));
copy.pressureKickScale=reference->pressureKickScale;std::memcpy(copy.pressureKickScale,s->pressureKickScale,1024*sizeof(*s->pressureKickScale));
copy.pressureDeltaMomX=reference->pressureDeltaMomX;std::memcpy(copy.pressureDeltaMomX,s->pressureDeltaMomX,1024*sizeof(*s->pressureDeltaMomX));
copy.pressureDeltaMomY=reference->pressureDeltaMomY;std::memcpy(copy.pressureDeltaMomY,s->pressureDeltaMomY,1024*sizeof(*s->pressureDeltaMomY));
copy.pressureDeltaMomZ=reference->pressureDeltaMomZ;std::memcpy(copy.pressureDeltaMomZ,s->pressureDeltaMomZ,1024*sizeof(*s->pressureDeltaMomZ));
copy.pressureDeltaEnergy=reference->pressureDeltaEnergy;std::memcpy(copy.pressureDeltaEnergy,s->pressureDeltaEnergy,1024*sizeof(*s->pressureDeltaEnergy));
copy.pressureParticleMoments=reference->pressureParticleMoments;std::memcpy(copy.pressureParticleMoments,s->pressureParticleMoments,1024*sizeof(*s->pressureParticleMoments));
copy.pressureParticleCount=reference->pressureParticleCount;std::memcpy(copy.pressureParticleCount,s->pressureParticleCount,1024*sizeof(*s->pressureParticleCount));
copy.solidPressurePhiMomX=reference->solidPressurePhiMomX;std::memcpy(copy.solidPressurePhiMomX,s->solidPressurePhiMomX,1024*sizeof(*s->solidPressurePhiMomX));
copy.solidPressurePhiMomY=reference->solidPressurePhiMomY;std::memcpy(copy.solidPressurePhiMomY,s->solidPressurePhiMomY,1024*sizeof(*s->solidPressurePhiMomY));
copy.solidPressurePhiMomZ=reference->solidPressurePhiMomZ;std::memcpy(copy.solidPressurePhiMomZ,s->solidPressurePhiMomZ,1024*sizeof(*s->solidPressurePhiMomZ));
copy.solidPressurePhiEnergy=reference->solidPressurePhiEnergy;std::memcpy(copy.solidPressurePhiEnergy,s->solidPressurePhiEnergy,1024*sizeof(*s->solidPressurePhiEnergy));
copy.mobilePackingRho=reference->mobilePackingRho;std::memcpy(copy.mobilePackingRho,s->mobilePackingRho,1024*sizeof(*s->mobilePackingRho));
copy.packingStuckRho=reference->packingStuckRho;std::memcpy(copy.packingStuckRho,s->packingStuckRho,1024*sizeof(*s->packingStuckRho));
copy.mobilePackingMomX=reference->mobilePackingMomX;std::memcpy(copy.mobilePackingMomX,s->mobilePackingMomX,1024*sizeof(*s->mobilePackingMomX));
copy.mobilePackingMomY=reference->mobilePackingMomY;std::memcpy(copy.mobilePackingMomY,s->mobilePackingMomY,1024*sizeof(*s->mobilePackingMomY));
copy.mobilePackingMomZ=reference->mobilePackingMomZ;std::memcpy(copy.mobilePackingMomZ,s->mobilePackingMomZ,1024*sizeof(*s->mobilePackingMomZ));
copy.mobilePackingActiveCellMask=reference->mobilePackingActiveCellMask;std::memcpy(copy.mobilePackingActiveCellMask,s->mobilePackingActiveCellMask,1024*sizeof(*s->mobilePackingActiveCellMask));
copy.mobilePackingCorrectionCellMask=reference->mobilePackingCorrectionCellMask;std::memcpy(copy.mobilePackingCorrectionCellMask,s->mobilePackingCorrectionCellMask,1024*sizeof(*s->mobilePackingCorrectionCellMask));
copy.mobilePackingActiveCellList=reference->mobilePackingActiveCellList;std::memcpy(copy.mobilePackingActiveCellList,s->mobilePackingActiveCellList,1024*sizeof(*s->mobilePackingActiveCellList));
copy.mobilePackingCorrectionCellList=reference->mobilePackingCorrectionCellList;std::memcpy(copy.mobilePackingCorrectionCellList,s->mobilePackingCorrectionCellList,1024*sizeof(*s->mobilePackingCorrectionCellList));
copy.mobilePackingFrontierCurrent=reference->mobilePackingFrontierCurrent;std::memcpy(copy.mobilePackingFrontierCurrent,s->mobilePackingFrontierCurrent,1024*sizeof(*s->mobilePackingFrontierCurrent));
copy.mobilePackingFrontierNext=reference->mobilePackingFrontierNext;std::memcpy(copy.mobilePackingFrontierNext,s->mobilePackingFrontierNext,1024*sizeof(*s->mobilePackingFrontierNext));
copy.mobilePackingActiveCellCount=reference->mobilePackingActiveCellCount;std::memcpy(copy.mobilePackingActiveCellCount,s->mobilePackingActiveCellCount,1024*sizeof(*s->mobilePackingActiveCellCount));
copy.mobilePackingCorrectionCellCount=reference->mobilePackingCorrectionCellCount;std::memcpy(copy.mobilePackingCorrectionCellCount,s->mobilePackingCorrectionCellCount,1024*sizeof(*s->mobilePackingCorrectionCellCount));
copy.mobilePackingFrontierCurrentCount=reference->mobilePackingFrontierCurrentCount;std::memcpy(copy.mobilePackingFrontierCurrentCount,s->mobilePackingFrontierCurrentCount,1024*sizeof(*s->mobilePackingFrontierCurrentCount));
copy.mobilePackingFrontierNextCount=reference->mobilePackingFrontierNextCount;std::memcpy(copy.mobilePackingFrontierNextCount,s->mobilePackingFrontierNextCount,1024*sizeof(*s->mobilePackingFrontierNextCount));
copy.thetaDragAlpha=reference->thetaDragAlpha;std::memcpy(copy.thetaDragAlpha,s->thetaDragAlpha,1024*sizeof(*s->thetaDragAlpha));
copy.momRhoP=reference->momRhoP;std::memcpy(copy.momRhoP,s->momRhoP,1024*sizeof(*s->momRhoP));
copy.momRhoUPx=reference->momRhoUPx;std::memcpy(copy.momRhoUPx,s->momRhoUPx,1024*sizeof(*s->momRhoUPx));
copy.momRhoUPy=reference->momRhoUPy;std::memcpy(copy.momRhoUPy,s->momRhoUPy,1024*sizeof(*s->momRhoUPy));
copy.momRhoUPz=reference->momRhoUPz;std::memcpy(copy.momRhoUPz,s->momRhoUPz,1024*sizeof(*s->momRhoUPz));
copy.momRhoEP=reference->momRhoEP;std::memcpy(copy.momRhoEP,s->momRhoEP,1024*sizeof(*s->momRhoEP));
copy.momRhoPD=reference->momRhoPD;std::memcpy(copy.momRhoPD,s->momRhoPD,1024*sizeof(*s->momRhoPD));
copy.momRhoHpP=reference->momRhoHpP;std::memcpy(copy.momRhoHpP,s->momRhoHpP,1024*sizeof(*s->momRhoHpP));
copy.radiationMobileMass=reference->radiationMobileMass;std::memcpy(copy.radiationMobileMass,s->radiationMobileMass,1024*sizeof(*s->radiationMobileMass));
copy.radiationMobileTemperatureMass=reference->radiationMobileTemperatureMass;std::memcpy(copy.radiationMobileTemperatureMass,s->radiationMobileTemperatureMass,1024*sizeof(*s->radiationMobileTemperatureMass));
copy.radiationMobileDiameterMass=reference->radiationMobileDiameterMass;std::memcpy(copy.radiationMobileDiameterMass,s->radiationMobileDiameterMass,1024*sizeof(*s->radiationMobileDiameterMass));
copy.poolThermalCount=reference->poolThermalCount;std::memcpy(copy.poolThermalCount,s->poolThermalCount,1024*sizeof(*s->poolThermalCount));
copy.poolThermalSumUx=reference->poolThermalSumUx;std::memcpy(copy.poolThermalSumUx,s->poolThermalSumUx,1024*sizeof(*s->poolThermalSumUx));
copy.poolThermalSumUy=reference->poolThermalSumUy;std::memcpy(copy.poolThermalSumUy,s->poolThermalSumUy,1024*sizeof(*s->poolThermalSumUy));
copy.poolThermalSumUz=reference->poolThermalSumUz;std::memcpy(copy.poolThermalSumUz,s->poolThermalSumUz,1024*sizeof(*s->poolThermalSumUz));
copy.poolThermalSumU2=reference->poolThermalSumU2;std::memcpy(copy.poolThermalSumU2,s->poolThermalSumU2,1024*sizeof(*s->poolThermalSumU2));
copy.poissonPoolSampleTargetCount=reference->poissonPoolSampleTargetCount;std::memcpy(copy.poissonPoolSampleTargetCount,s->poissonPoolSampleTargetCount,1024*sizeof(*s->poissonPoolSampleTargetCount));
copy.poissonPoolMass=reference->poissonPoolMass;std::memcpy(copy.poissonPoolMass,s->poissonPoolMass,1024*sizeof(*s->poissonPoolMass));
copy.poissonPoolMomX=reference->poissonPoolMomX;std::memcpy(copy.poissonPoolMomX,s->poissonPoolMomX,1024*sizeof(*s->poissonPoolMomX));
copy.poissonPoolMomY=reference->poissonPoolMomY;std::memcpy(copy.poissonPoolMomY,s->poissonPoolMomY,1024*sizeof(*s->poissonPoolMomY));
copy.poissonPoolMomZ=reference->poissonPoolMomZ;std::memcpy(copy.poissonPoolMomZ,s->poissonPoolMomZ,1024*sizeof(*s->poissonPoolMomZ));
copy.poissonPoolEnergy=reference->poissonPoolEnergy;std::memcpy(copy.poissonPoolEnergy,s->poissonPoolEnergy,1024*sizeof(*s->poissonPoolEnergy));
copy.poissonPoolDiameter=reference->poissonPoolDiameter;std::memcpy(copy.poissonPoolDiameter,s->poissonPoolDiameter,1024*sizeof(*s->poissonPoolDiameter));
copy.poissonPoolDiameter2=reference->poissonPoolDiameter2;std::memcpy(copy.poissonPoolDiameter2,s->poissonPoolDiameter2,1024*sizeof(*s->poissonPoolDiameter2));
copy.px=reference->px;std::memcpy(copy.px,s->px,1024*sizeof(*s->px));
copy.py=reference->py;std::memcpy(copy.py,s->py,1024*sizeof(*s->py));
copy.pz=reference->pz;std::memcpy(copy.pz,s->pz,1024*sizeof(*s->pz));
copy.pux=reference->pux;std::memcpy(copy.pux,s->pux,1024*sizeof(*s->pux));
copy.puy=reference->puy;std::memcpy(copy.puy,s->puy,1024*sizeof(*s->puy));
copy.puz=reference->puz;std::memcpy(copy.puz,s->puz,1024*sizeof(*s->puz));
copy.puxOld=reference->puxOld;std::memcpy(copy.puxOld,s->puxOld,1024*sizeof(*s->puxOld));
copy.puyOld=reference->puyOld;std::memcpy(copy.puyOld,s->puyOld,1024*sizeof(*s->puyOld));
copy.puzOld=reference->puzOld;std::memcpy(copy.puzOld,s->puzOld,1024*sizeof(*s->puzOld));
copy.pT=reference->pT;std::memcpy(copy.pT,s->pT,1024*sizeof(*s->pT));
copy.pThermalEnthalpyResidual=reference->pThermalEnthalpyResidual;std::memcpy(copy.pThermalEnthalpyResidual,s->pThermalEnthalpyResidual,1024*sizeof(*s->pThermalEnthalpyResidual));
copy.pTheta=reference->pTheta;std::memcpy(copy.pTheta,s->pTheta,1024*sizeof(*s->pTheta));
copy.pContactAge=reference->pContactAge;std::memcpy(copy.pContactAge,s->pContactAge,1024*sizeof(*s->pContactAge));
copy.pd=reference->pd;std::memcpy(copy.pd,s->pd,1024*sizeof(*s->pd));
copy.pm=reference->pm;std::memcpy(copy.pm,s->pm,1024*sizeof(*s->pm));
copy.pCellId=reference->pCellId;std::memcpy(copy.pCellId,s->pCellId,1024*sizeof(*s->pCellId));
copy.pStatus=reference->pStatus;std::memcpy(copy.pStatus,s->pStatus,1024*sizeof(*s->pStatus));
copy.pStuck=reference->pStuck;std::memcpy(copy.pStuck,s->pStuck,1024*sizeof(*s->pStuck));
copy.pStuckFaceId=reference->pStuckFaceId;std::memcpy(copy.pStuckFaceId,s->pStuckFaceId,1024*sizeof(*s->pStuckFaceId));
copy.pDepositionArea=reference->pDepositionArea;std::memcpy(copy.pDepositionArea,s->pDepositionArea,1024*sizeof(*s->pDepositionArea));
copy.pContactDuration=reference->pContactDuration;std::memcpy(copy.pContactDuration,s->pContactDuration,1024*sizeof(*s->pContactDuration));
copy.pContactMaximumArea=reference->pContactMaximumArea;std::memcpy(copy.pContactMaximumArea,s->pContactMaximumArea,1024*sizeof(*s->pContactMaximumArea));
copy.pContactPeakFraction=reference->pContactPeakFraction;std::memcpy(copy.pContactPeakFraction,s->pContactPeakFraction,1024*sizeof(*s->pContactPeakFraction));
copy.pColdNodeSpecificEnthalpy=reference->pColdNodeSpecificEnthalpy;std::memcpy(copy.pColdNodeSpecificEnthalpy,s->pColdNodeSpecificEnthalpy,1024*sizeof(*s->pColdNodeSpecificEnthalpy));
copy.pColdRingSolidMass=reference->pColdRingSolidMass;std::memcpy(copy.pColdRingSolidMass,s->pColdRingSolidMass,1024*sizeof(*s->pColdRingSolidMass));
copy.pColdFrozenArea=reference->pColdFrozenArea;std::memcpy(copy.pColdFrozenArea,s->pColdFrozenArea,1024*sizeof(*s->pColdFrozenArea));
copy.pColdContactAge=reference->pColdContactAge;std::memcpy(copy.pColdContactAge,s->pColdContactAge,1024*sizeof(*s->pColdContactAge));
copy.pCold2DNodeSpecificEnthalpy=reference->pCold2DNodeSpecificEnthalpy;std::memcpy(copy.pCold2DNodeSpecificEnthalpy,s->pCold2DNodeSpecificEnthalpy,1024*sizeof(*s->pCold2DNodeSpecificEnthalpy));
copy.pCold2DRingContactAge=reference->pCold2DRingContactAge;std::memcpy(copy.pCold2DRingContactAge,s->pCold2DRingContactAge,1024*sizeof(*s->pCold2DRingContactAge));
copy.pCold2DFrozenArea=reference->pCold2DFrozenArea;std::memcpy(copy.pCold2DFrozenArea,s->pCold2DFrozenArea,1024*sizeof(*s->pCold2DFrozenArea));
copy.pRng=reference->pRng;std::memcpy(copy.pRng,s->pRng,1024*sizeof(*s->pRng));
copy.pOrigId=reference->pOrigId;std::memcpy(copy.pOrigId,s->pOrigId,1024*sizeof(*s->pOrigId));
copy.wallBoundParticleIndex=reference->wallBoundParticleIndex;std::memcpy(copy.wallBoundParticleIndex,s->wallBoundParticleIndex,1024*sizeof(*s->wallBoundParticleIndex));
copy.wallBoundParticleCountDevice=reference->wallBoundParticleCountDevice;std::memcpy(copy.wallBoundParticleCountDevice,s->wallBoundParticleCountDevice,1024*sizeof(*s->wallBoundParticleCountDevice));
copy.sortedParticleIndex=reference->sortedParticleIndex;std::memcpy(copy.sortedParticleIndex,s->sortedParticleIndex,1024*sizeof(*s->sortedParticleIndex));
copy.cellParticleCount=reference->cellParticleCount;std::memcpy(copy.cellParticleCount,s->cellParticleCount,1024*sizeof(*s->cellParticleCount));
copy.cellParticleOffset=reference->cellParticleOffset;std::memcpy(copy.cellParticleOffset,s->cellParticleOffset,1024*sizeof(*s->cellParticleOffset));
copy.compactCellOffset=reference->compactCellOffset;std::memcpy(copy.compactCellOffset,s->compactCellOffset,1024*sizeof(*s->compactCellOffset));
copy.cellParticleWrite=reference->cellParticleWrite;std::memcpy(copy.cellParticleWrite,s->cellParticleWrite,1024*sizeof(*s->cellParticleWrite));
copy.preBaseCellOffset=reference->preBaseCellOffset;std::memcpy(copy.preBaseCellOffset,s->preBaseCellOffset,1024*sizeof(*s->preBaseCellOffset));
copy.preBaseParticleCountDevice=reference->preBaseParticleCountDevice;std::memcpy(copy.preBaseParticleCountDevice,s->preBaseParticleCountDevice,1024*sizeof(*s->preBaseParticleCountDevice));
copy.cellScanTempStorage=reference->cellScanTempStorage;std::memcpy(copy.cellScanTempStorage,s->cellScanTempStorage,1024*sizeof(*s->cellScanTempStorage));
copy.compactCountDevice=reference->compactCountDevice;std::memcpy(copy.compactCountDevice,s->compactCountDevice,1024*sizeof(*s->compactCountDevice));
copy.compactSelectTempStorage=reference->compactSelectTempStorage;std::memcpy(copy.compactSelectTempStorage,s->compactSelectTempStorage,1024*sizeof(*s->compactSelectTempStorage));
copy.csrCellTaskCount=reference->csrCellTaskCount;std::memcpy(copy.csrCellTaskCount,s->csrCellTaskCount,1024*sizeof(*s->csrCellTaskCount));
copy.csrCellTaskOffset=reference->csrCellTaskOffset;std::memcpy(copy.csrCellTaskOffset,s->csrCellTaskOffset,1024*sizeof(*s->csrCellTaskOffset));
copy.csrMultiTaskCellList=reference->csrMultiTaskCellList;std::memcpy(copy.csrMultiTaskCellList,s->csrMultiTaskCellList,1024*sizeof(*s->csrMultiTaskCellList));
copy.csrHeavyCellList=reference->csrHeavyCellList;std::memcpy(copy.csrHeavyCellList,s->csrHeavyCellList,1024*sizeof(*s->csrHeavyCellList));
copy.csrHeavyTaskCell=reference->csrHeavyTaskCell;std::memcpy(copy.csrHeavyTaskCell,s->csrHeavyTaskCell,1024*sizeof(*s->csrHeavyTaskCell));
copy.csrHeavyTaskBegin=reference->csrHeavyTaskBegin;std::memcpy(copy.csrHeavyTaskBegin,s->csrHeavyTaskBegin,1024*sizeof(*s->csrHeavyTaskBegin));
copy.csrHeavyTaskEnd=reference->csrHeavyTaskEnd;std::memcpy(copy.csrHeavyTaskEnd,s->csrHeavyTaskEnd,1024*sizeof(*s->csrHeavyTaskEnd));
copy.csrHeavyCellTaskStart=reference->csrHeavyCellTaskStart;std::memcpy(copy.csrHeavyCellTaskStart,s->csrHeavyCellTaskStart,1024*sizeof(*s->csrHeavyCellTaskStart));
copy.csrHeavyCellTaskCount=reference->csrHeavyCellTaskCount;std::memcpy(copy.csrHeavyCellTaskCount,s->csrHeavyCellTaskCount,1024*sizeof(*s->csrHeavyCellTaskCount));
copy.csrHeavyPartials=reference->csrHeavyPartials;std::memcpy(copy.csrHeavyPartials,s->csrHeavyPartials,1024*sizeof(*s->csrHeavyPartials));
copy.csrHeavyInjectionTaskCount=reference->csrHeavyInjectionTaskCount;std::memcpy(copy.csrHeavyInjectionTaskCount,s->csrHeavyInjectionTaskCount,1024*sizeof(*s->csrHeavyInjectionTaskCount));
copy.csrHeavyInjectionTaskCursor=reference->csrHeavyInjectionTaskCursor;std::memcpy(copy.csrHeavyInjectionTaskCursor,s->csrHeavyInjectionTaskCursor,1024*sizeof(*s->csrHeavyInjectionTaskCursor));
copy.csrHeavyInjectionTaskCell=reference->csrHeavyInjectionTaskCell;std::memcpy(copy.csrHeavyInjectionTaskCell,s->csrHeavyInjectionTaskCell,1024*sizeof(*s->csrHeavyInjectionTaskCell));
copy.csrHeavyInjectionTaskBegin=reference->csrHeavyInjectionTaskBegin;std::memcpy(copy.csrHeavyInjectionTaskBegin,s->csrHeavyInjectionTaskBegin,1024*sizeof(*s->csrHeavyInjectionTaskBegin));
copy.csrHeavyInjectionTaskEnd=reference->csrHeavyInjectionTaskEnd;std::memcpy(copy.csrHeavyInjectionTaskEnd,s->csrHeavyInjectionTaskEnd,1024*sizeof(*s->csrHeavyInjectionTaskEnd));
copy.csrHeavyInjectionCellTaskStart=reference->csrHeavyInjectionCellTaskStart;std::memcpy(copy.csrHeavyInjectionCellTaskStart,s->csrHeavyInjectionCellTaskStart,1024*sizeof(*s->csrHeavyInjectionCellTaskStart));
copy.csrHeavyInjectionCellTaskCount=reference->csrHeavyInjectionCellTaskCount;std::memcpy(copy.csrHeavyInjectionCellTaskCount,s->csrHeavyInjectionCellTaskCount,1024*sizeof(*s->csrHeavyInjectionCellTaskCount));
copy.csrHeavyInjectionPartials=reference->csrHeavyInjectionPartials;std::memcpy(copy.csrHeavyInjectionPartials,s->csrHeavyInjectionPartials,1024*sizeof(*s->csrHeavyInjectionPartials));
copy.sourceCell=reference->sourceCell;std::memcpy(copy.sourceCell,s->sourceCell,1024*sizeof(*s->sourceCell));
copy.sourceFace=reference->sourceFace;std::memcpy(copy.sourceFace,s->sourceFace,1024*sizeof(*s->sourceFace));
copy.sourcePx=reference->sourcePx;std::memcpy(copy.sourcePx,s->sourcePx,1024*sizeof(*s->sourcePx));
copy.sourcePy=reference->sourcePy;std::memcpy(copy.sourcePy,s->sourcePy,1024*sizeof(*s->sourcePy));
copy.sourcePz=reference->sourcePz;std::memcpy(copy.sourcePz,s->sourcePz,1024*sizeof(*s->sourcePz));
copy.sourceUx=reference->sourceUx;std::memcpy(copy.sourceUx,s->sourceUx,1024*sizeof(*s->sourceUx));
copy.sourceUy=reference->sourceUy;std::memcpy(copy.sourceUy,s->sourceUy,1024*sizeof(*s->sourceUy));
copy.sourceUz=reference->sourceUz;std::memcpy(copy.sourceUz,s->sourceUz,1024*sizeof(*s->sourceUz));
copy.sourceT=reference->sourceT;std::memcpy(copy.sourceT,s->sourceT,1024*sizeof(*s->sourceT));
copy.sourceTheta=reference->sourceTheta;std::memcpy(copy.sourceTheta,s->sourceTheta,1024*sizeof(*s->sourceTheta));
copy.sourceD=reference->sourceD;std::memcpy(copy.sourceD,s->sourceD,1024*sizeof(*s->sourceD));
copy.sourceMassRate=reference->sourceMassRate;std::memcpy(copy.sourceMassRate,s->sourceMassRate,1024*sizeof(*s->sourceMassRate));
copy.sourceResidualMass=reference->sourceResidualMass;std::memcpy(copy.sourceResidualMass,s->sourceResidualMass,1024*sizeof(*s->sourceResidualMass));
copy.compactPx=reference->compactPx;std::memcpy(copy.compactPx,s->compactPx,1024*sizeof(*s->compactPx));
copy.compactPy=reference->compactPy;std::memcpy(copy.compactPy,s->compactPy,1024*sizeof(*s->compactPy));
copy.compactPz=reference->compactPz;std::memcpy(copy.compactPz,s->compactPz,1024*sizeof(*s->compactPz));
copy.compactPux=reference->compactPux;std::memcpy(copy.compactPux,s->compactPux,1024*sizeof(*s->compactPux));
copy.compactPuxOld=reference->compactPuxOld;std::memcpy(copy.compactPuxOld,s->compactPuxOld,1024*sizeof(*s->compactPuxOld));
copy.compactPuy=reference->compactPuy;std::memcpy(copy.compactPuy,s->compactPuy,1024*sizeof(*s->compactPuy));
copy.compactPuyOld=reference->compactPuyOld;std::memcpy(copy.compactPuyOld,s->compactPuyOld,1024*sizeof(*s->compactPuyOld));
copy.compactPuz=reference->compactPuz;std::memcpy(copy.compactPuz,s->compactPuz,1024*sizeof(*s->compactPuz));
copy.compactPuzOld=reference->compactPuzOld;std::memcpy(copy.compactPuzOld,s->compactPuzOld,1024*sizeof(*s->compactPuzOld));
copy.compactPT=reference->compactPT;std::memcpy(copy.compactPT,s->compactPT,1024*sizeof(*s->compactPT));
copy.compactPThermalEnthalpyResidual=reference->compactPThermalEnthalpyResidual;std::memcpy(copy.compactPThermalEnthalpyResidual,s->compactPThermalEnthalpyResidual,1024*sizeof(*s->compactPThermalEnthalpyResidual));
copy.compactPTheta=reference->compactPTheta;std::memcpy(copy.compactPTheta,s->compactPTheta,1024*sizeof(*s->compactPTheta));
copy.compactPContactAge=reference->compactPContactAge;std::memcpy(copy.compactPContactAge,s->compactPContactAge,1024*sizeof(*s->compactPContactAge));
copy.compactPd=reference->compactPd;std::memcpy(copy.compactPd,s->compactPd,1024*sizeof(*s->compactPd));
copy.compactPm=reference->compactPm;std::memcpy(copy.compactPm,s->compactPm,1024*sizeof(*s->compactPm));
copy.compactPCellId=reference->compactPCellId;std::memcpy(copy.compactPCellId,s->compactPCellId,1024*sizeof(*s->compactPCellId));
copy.compactPStatus=reference->compactPStatus;std::memcpy(copy.compactPStatus,s->compactPStatus,1024*sizeof(*s->compactPStatus));
copy.compactPStuck=reference->compactPStuck;std::memcpy(copy.compactPStuck,s->compactPStuck,1024*sizeof(*s->compactPStuck));
copy.compactPStuckFaceId=reference->compactPStuckFaceId;std::memcpy(copy.compactPStuckFaceId,s->compactPStuckFaceId,1024*sizeof(*s->compactPStuckFaceId));
copy.compactPDepositionArea=reference->compactPDepositionArea;std::memcpy(copy.compactPDepositionArea,s->compactPDepositionArea,1024*sizeof(*s->compactPDepositionArea));
copy.compactPContactDuration=reference->compactPContactDuration;std::memcpy(copy.compactPContactDuration,s->compactPContactDuration,1024*sizeof(*s->compactPContactDuration));
copy.compactPContactMaximumArea=reference->compactPContactMaximumArea;std::memcpy(copy.compactPContactMaximumArea,s->compactPContactMaximumArea,1024*sizeof(*s->compactPContactMaximumArea));
copy.compactPContactPeakFraction=reference->compactPContactPeakFraction;std::memcpy(copy.compactPContactPeakFraction,s->compactPContactPeakFraction,1024*sizeof(*s->compactPContactPeakFraction));
copy.compactPColdNodeSpecificEnthalpy=reference->compactPColdNodeSpecificEnthalpy;std::memcpy(copy.compactPColdNodeSpecificEnthalpy,s->compactPColdNodeSpecificEnthalpy,1024*sizeof(*s->compactPColdNodeSpecificEnthalpy));
copy.compactPColdRingSolidMass=reference->compactPColdRingSolidMass;std::memcpy(copy.compactPColdRingSolidMass,s->compactPColdRingSolidMass,1024*sizeof(*s->compactPColdRingSolidMass));
copy.compactPColdFrozenArea=reference->compactPColdFrozenArea;std::memcpy(copy.compactPColdFrozenArea,s->compactPColdFrozenArea,1024*sizeof(*s->compactPColdFrozenArea));
copy.compactPColdContactAge=reference->compactPColdContactAge;std::memcpy(copy.compactPColdContactAge,s->compactPColdContactAge,1024*sizeof(*s->compactPColdContactAge));
copy.compactPCold2DNodeSpecificEnthalpy=reference->compactPCold2DNodeSpecificEnthalpy;std::memcpy(copy.compactPCold2DNodeSpecificEnthalpy,s->compactPCold2DNodeSpecificEnthalpy,1024*sizeof(*s->compactPCold2DNodeSpecificEnthalpy));
copy.compactPCold2DRingContactAge=reference->compactPCold2DRingContactAge;std::memcpy(copy.compactPCold2DRingContactAge,s->compactPCold2DRingContactAge,1024*sizeof(*s->compactPCold2DRingContactAge));
copy.compactPCold2DFrozenArea=reference->compactPCold2DFrozenArea;std::memcpy(copy.compactPCold2DFrozenArea,s->compactPCold2DFrozenArea,1024*sizeof(*s->compactPCold2DFrozenArea));
copy.compactPRng=reference->compactPRng;std::memcpy(copy.compactPRng,s->compactPRng,1024*sizeof(*s->compactPRng));
copy.compactPOrigId=reference->compactPOrigId;std::memcpy(copy.compactPOrigId,s->compactPOrigId,1024*sizeof(*s->compactPOrigId));
*reference=copy;}
applyCollisionalPressureProjectionCellAtomicKernel<<<1,32>>>(s,.000100000000000003);baselinePressureCell<<<1,32>>>(reference,.000100000000000003);deviceSync();for(int c=0;c<16;c++)equal(s->pressureDeltaMomX[c],reference->pressureDeltaMomX[c],"pressure pressureDeltaMomX");
for(int c=0;c<16;c++)equal(s->pressureDeltaMomY[c],reference->pressureDeltaMomY[c],"pressure pressureDeltaMomY");
for(int c=0;c<16;c++)equal(s->pressureDeltaMomZ[c],reference->pressureDeltaMomZ[c],"pressure pressureDeltaMomZ");
for(int c=0;c<16;c++)equal(s->pressureDeltaEnergy[c],reference->pressureDeltaEnergy[c],"pressure pressureDeltaEnergy");
for(int c=0;c<16;c++)equal(s->momRhoUPx[c],reference->momRhoUPx[c],"pressure momRhoUPx");
for(int c=0;c<16;c++)equal(s->momRhoUPy[c],reference->momRhoUPy[c],"pressure momRhoUPy");
for(int c=0;c<16;c++)equal(s->momRhoUPz[c],reference->momRhoUPz[c],"pressure momRhoUPz");
for(int c=0;c<16;c++)equal(s->momRhoEP[c],reference->momRhoEP[c],"pressure momRhoEP");
for(int c=0;c<16;c++)equal(s->rhoUsx[c],reference->rhoUsx[c],"pressure rhoUsx");
for(int c=0;c<16;c++)equal(s->rhoUsy[c],reference->rhoUsy[c],"pressure rhoUsy");
for(int c=0;c<16;c++)equal(s->rhoUsz[c],reference->rhoUsz[c],"pressure rhoUsz");
for(int c=0;c<16;c++)equal(s->rhoEs[c],reference->rhoEs[c],"pressure rhoEs");
for(int c=0;c<16;c++)equal(s->Usx[c],reference->Usx[c],"pressure Usx");
for(int c=0;c<16;c++)equal(s->Usy[c],reference->Usy[c],"pressure Usy");
for(int c=0;c<16;c++)equal(s->Usz[c],reference->Usz[c],"pressure Usz");
for(int c=0;c<16;c++)equal(s->theta[c],reference->theta[c],"pressure theta");

s->particleCapacity=64;*s->particleCountDevice=64;
for(int i=0;i<64;i++){
 s->pStatus[i]=1;s->pCellId[i]=i%16;s->pm[i]=.5;s->pd[i]=1e-4;s->pT[i]=1000;s->pTheta[i]=.25;
 s->pux[i]=.25+(i%3)*.125;s->puy[i]=-.125;s->puz[i]=0;s->puxOld[i]=2;s->puyOld[i]=-1;s->puzOld[i]=.5;
 s->pStuck[i]=(i/16==0)?Foam::gpuThermal::particleWallMobile:(i/16==1)?Foam::gpuThermal::particleWallTransientRebound:Foam::gpuThermal::particleWallDeposited;
}
{DeviceState copy=*s;
copy.csrMaximumOccupancy=reference->csrMaximumOccupancy;std::memcpy(copy.csrMaximumOccupancy,s->csrMaximumOccupancy,1024*sizeof(*s->csrMaximumOccupancy));
copy.csrHeavyTaskCount=reference->csrHeavyTaskCount;std::memcpy(copy.csrHeavyTaskCount,s->csrHeavyTaskCount,1024*sizeof(*s->csrHeavyTaskCount));
copy.csrHeavyCellCount=reference->csrHeavyCellCount;std::memcpy(copy.csrHeavyCellCount,s->csrHeavyCellCount,1024*sizeof(*s->csrHeavyCellCount));
copy.csrHeavyTaskCursor=reference->csrHeavyTaskCursor;std::memcpy(copy.csrHeavyTaskCursor,s->csrHeavyTaskCursor,1024*sizeof(*s->csrHeavyTaskCursor));
copy.particleCountDevice=reference->particleCountDevice;std::memcpy(copy.particleCountDevice,s->particleCountDevice,1024*sizeof(*s->particleCountDevice));
copy.faceOwner=reference->faceOwner;std::memcpy(copy.faceOwner,s->faceOwner,1024*sizeof(*s->faceOwner));
copy.faceNeighbour=reference->faceNeighbour;std::memcpy(copy.faceNeighbour,s->faceNeighbour,1024*sizeof(*s->faceNeighbour));
copy.facePeriodicPair=reference->facePeriodicPair;std::memcpy(copy.facePeriodicPair,s->facePeriodicPair,1024*sizeof(*s->facePeriodicPair));
copy.facePeriodicDx=reference->facePeriodicDx;std::memcpy(copy.facePeriodicDx,s->facePeriodicDx,1024*sizeof(*s->facePeriodicDx));
copy.facePeriodicDy=reference->facePeriodicDy;std::memcpy(copy.facePeriodicDy,s->facePeriodicDy,1024*sizeof(*s->facePeriodicDy));
copy.facePeriodicDz=reference->facePeriodicDz;std::memcpy(copy.facePeriodicDz,s->facePeriodicDz,1024*sizeof(*s->facePeriodicDz));
copy.V=reference->V;std::memcpy(copy.V,s->V,1024*sizeof(*s->V));
copy.Cx=reference->Cx;std::memcpy(copy.Cx,s->Cx,1024*sizeof(*s->Cx));
copy.Cy=reference->Cy;std::memcpy(copy.Cy,s->Cy,1024*sizeof(*s->Cy));
copy.Cz=reference->Cz;std::memcpy(copy.Cz,s->Cz,1024*sizeof(*s->Cz));
copy.faceCx=reference->faceCx;std::memcpy(copy.faceCx,s->faceCx,1024*sizeof(*s->faceCx));
copy.faceCy=reference->faceCy;std::memcpy(copy.faceCy,s->faceCy,1024*sizeof(*s->faceCy));
copy.faceCz=reference->faceCz;std::memcpy(copy.faceCz,s->faceCz,1024*sizeof(*s->faceCz));
copy.Sfx=reference->Sfx;std::memcpy(copy.Sfx,s->Sfx,1024*sizeof(*s->Sfx));
copy.Sfy=reference->Sfy;std::memcpy(copy.Sfy,s->Sfy,1024*sizeof(*s->Sfy));
copy.Sfz=reference->Sfz;std::memcpy(copy.Sfz,s->Sfz,1024*sizeof(*s->Sfz));
copy.magSf=reference->magSf;std::memcpy(copy.magSf,s->magSf,1024*sizeof(*s->magSf));
copy.deltaCoeffs=reference->deltaCoeffs;std::memcpy(copy.deltaCoeffs,s->deltaCoeffs,1024*sizeof(*s->deltaCoeffs));
copy.faceWeight=reference->faceWeight;std::memcpy(copy.faceWeight,s->faceWeight,1024*sizeof(*s->faceWeight));
copy.cellLength=reference->cellLength;std::memcpy(copy.cellLength,s->cellLength,1024*sizeof(*s->cellLength));
copy.cellPlaneStart=reference->cellPlaneStart;std::memcpy(copy.cellPlaneStart,s->cellPlaneStart,1024*sizeof(*s->cellPlaneStart));
copy.cellPlaneCount=reference->cellPlaneCount;std::memcpy(copy.cellPlaneCount,s->cellPlaneCount,1024*sizeof(*s->cellPlaneCount));
copy.cellFaceId=reference->cellFaceId;std::memcpy(copy.cellFaceId,s->cellFaceId,1024*sizeof(*s->cellFaceId));
copy.cellFaceNeighbor=reference->cellFaceNeighbor;std::memcpy(copy.cellFaceNeighbor,s->cellFaceNeighbor,1024*sizeof(*s->cellFaceNeighbor));
copy.cellFaceKind=reference->cellFaceKind;std::memcpy(copy.cellFaceKind,s->cellFaceKind,1024*sizeof(*s->cellFaceKind));
copy.cellFaceRestitution=reference->cellFaceRestitution;std::memcpy(copy.cellFaceRestitution,s->cellFaceRestitution,1024*sizeof(*s->cellFaceRestitution));
copy.cellFaceTangential=reference->cellFaceTangential;std::memcpy(copy.cellFaceTangential,s->cellFaceTangential,1024*sizeof(*s->cellFaceTangential));
copy.planeNx=reference->planeNx;std::memcpy(copy.planeNx,s->planeNx,1024*sizeof(*s->planeNx));
copy.planeNy=reference->planeNy;std::memcpy(copy.planeNy,s->planeNy,1024*sizeof(*s->planeNy));
copy.planeNz=reference->planeNz;std::memcpy(copy.planeNz,s->planeNz,1024*sizeof(*s->planeNz));
copy.planeD=reference->planeD;std::memcpy(copy.planeD,s->planeD,1024*sizeof(*s->planeD));
copy.rho=reference->rho;std::memcpy(copy.rho,s->rho,1024*sizeof(*s->rho));
copy.rhoUx=reference->rhoUx;std::memcpy(copy.rhoUx,s->rhoUx,1024*sizeof(*s->rhoUx));
copy.rhoUy=reference->rhoUy;std::memcpy(copy.rhoUy,s->rhoUy,1024*sizeof(*s->rhoUy));
copy.rhoUz=reference->rhoUz;std::memcpy(copy.rhoUz,s->rhoUz,1024*sizeof(*s->rhoUz));
copy.rhoE=reference->rhoE;std::memcpy(copy.rhoE,s->rhoE,1024*sizeof(*s->rhoE));
copy.rhoNext=reference->rhoNext;std::memcpy(copy.rhoNext,s->rhoNext,1024*sizeof(*s->rhoNext));
copy.rhoUxNext=reference->rhoUxNext;std::memcpy(copy.rhoUxNext,s->rhoUxNext,1024*sizeof(*s->rhoUxNext));
copy.rhoUyNext=reference->rhoUyNext;std::memcpy(copy.rhoUyNext,s->rhoUyNext,1024*sizeof(*s->rhoUyNext));
copy.rhoUzNext=reference->rhoUzNext;std::memcpy(copy.rhoUzNext,s->rhoUzNext,1024*sizeof(*s->rhoUzNext));
copy.rhoENext=reference->rhoENext;std::memcpy(copy.rhoENext,s->rhoENext,1024*sizeof(*s->rhoENext));
copy.gasPhiRho=reference->gasPhiRho;std::memcpy(copy.gasPhiRho,s->gasPhiRho,1024*sizeof(*s->gasPhiRho));
copy.gasPhiRhoUx=reference->gasPhiRhoUx;std::memcpy(copy.gasPhiRhoUx,s->gasPhiRhoUx,1024*sizeof(*s->gasPhiRhoUx));
copy.gasPhiRhoUy=reference->gasPhiRhoUy;std::memcpy(copy.gasPhiRhoUy,s->gasPhiRhoUy,1024*sizeof(*s->gasPhiRhoUy));
copy.gasPhiRhoUz=reference->gasPhiRhoUz;std::memcpy(copy.gasPhiRhoUz,s->gasPhiRhoUz,1024*sizeof(*s->gasPhiRhoUz));
copy.gasPhiRhoE=reference->gasPhiRhoE;std::memcpy(copy.gasPhiRhoE,s->gasPhiRhoE,1024*sizeof(*s->gasPhiRhoE));
copy.gasFluxPositivityScale=reference->gasFluxPositivityScale;std::memcpy(copy.gasFluxPositivityScale,s->gasFluxPositivityScale,1024*sizeof(*s->gasFluxPositivityScale));
copy.gasHllcAdcSensor=reference->gasHllcAdcSensor;std::memcpy(copy.gasHllcAdcSensor,s->gasHllcAdcSensor,1024*sizeof(*s->gasHllcAdcSensor));
copy.particleStuckCandidateMask=reference->particleStuckCandidateMask;std::memcpy(copy.particleStuckCandidateMask,s->particleStuckCandidateMask,1024*sizeof(*s->particleStuckCandidateMask));
copy.particleWallDepositedEnergy=reference->particleWallDepositedEnergy;std::memcpy(copy.particleWallDepositedEnergy,s->particleWallDepositedEnergy,1024*sizeof(*s->particleWallDepositedEnergy));
copy.particleWallReflectedEnergy=reference->particleWallReflectedEnergy;std::memcpy(copy.particleWallReflectedEnergy,s->particleWallReflectedEnergy,1024*sizeof(*s->particleWallReflectedEnergy));
copy.particleWallRepresentedContactArea=reference->particleWallRepresentedContactArea;std::memcpy(copy.particleWallRepresentedContactArea,s->particleWallRepresentedContactArea,1024*sizeof(*s->particleWallRepresentedContactArea));
copy.particleWallContactAreaScale=reference->particleWallContactAreaScale;std::memcpy(copy.particleWallContactAreaScale,s->particleWallContactAreaScale,1024*sizeof(*s->particleWallContactAreaScale));
copy.particleWallEffusivityByFace=reference->particleWallEffusivityByFace;std::memcpy(copy.particleWallEffusivityByFace,s->particleWallEffusivityByFace,1024*sizeof(*s->particleWallEffusivityByFace));
copy.finiteContactRateTable=reference->finiteContactRateTable;std::memcpy(copy.finiteContactRateTable,s->finiteContactRateTable,1024*sizeof(*s->finiteContactRateTable));
copy.gasBoundaryKind=reference->gasBoundaryKind;std::memcpy(copy.gasBoundaryKind,s->gasBoundaryKind,1024*sizeof(*s->gasBoundaryKind));
copy.gasBoundaryRhoFix=reference->gasBoundaryRhoFix;std::memcpy(copy.gasBoundaryRhoFix,s->gasBoundaryRhoFix,1024*sizeof(*s->gasBoundaryRhoFix));
copy.gasBoundaryUFix=reference->gasBoundaryUFix;std::memcpy(copy.gasBoundaryUFix,s->gasBoundaryUFix,1024*sizeof(*s->gasBoundaryUFix));
copy.gasBoundaryPFix=reference->gasBoundaryPFix;std::memcpy(copy.gasBoundaryPFix,s->gasBoundaryPFix,1024*sizeof(*s->gasBoundaryPFix));
copy.gasBoundaryTFix=reference->gasBoundaryTFix;std::memcpy(copy.gasBoundaryTFix,s->gasBoundaryTFix,1024*sizeof(*s->gasBoundaryTFix));
copy.gasBoundaryPWave=reference->gasBoundaryPWave;std::memcpy(copy.gasBoundaryPWave,s->gasBoundaryPWave,1024*sizeof(*s->gasBoundaryPWave));
copy.gasBoundaryPWaveGamma=reference->gasBoundaryPWaveGamma;std::memcpy(copy.gasBoundaryPWaveGamma,s->gasBoundaryPWaveGamma,1024*sizeof(*s->gasBoundaryPWaveGamma));
copy.gasBoundaryPWaveFieldInf=reference->gasBoundaryPWaveFieldInf;std::memcpy(copy.gasBoundaryPWaveFieldInf,s->gasBoundaryPWaveFieldInf,1024*sizeof(*s->gasBoundaryPWaveFieldInf));
copy.gasBoundaryPWaveLInf=reference->gasBoundaryPWaveLInf;std::memcpy(copy.gasBoundaryPWaveLInf,s->gasBoundaryPWaveLInf,1024*sizeof(*s->gasBoundaryPWaveLInf));
copy.gasBoundaryRho=reference->gasBoundaryRho;std::memcpy(copy.gasBoundaryRho,s->gasBoundaryRho,1024*sizeof(*s->gasBoundaryRho));
copy.gasBoundaryUx=reference->gasBoundaryUx;std::memcpy(copy.gasBoundaryUx,s->gasBoundaryUx,1024*sizeof(*s->gasBoundaryUx));
copy.gasBoundaryUy=reference->gasBoundaryUy;std::memcpy(copy.gasBoundaryUy,s->gasBoundaryUy,1024*sizeof(*s->gasBoundaryUy));
copy.gasBoundaryUz=reference->gasBoundaryUz;std::memcpy(copy.gasBoundaryUz,s->gasBoundaryUz,1024*sizeof(*s->gasBoundaryUz));
copy.gasBoundaryP=reference->gasBoundaryP;std::memcpy(copy.gasBoundaryP,s->gasBoundaryP,1024*sizeof(*s->gasBoundaryP));
copy.gasBoundaryT=reference->gasBoundaryT;std::memcpy(copy.gasBoundaryT,s->gasBoundaryT,1024*sizeof(*s->gasBoundaryT));
copy.riemannBoundaryKind=reference->riemannBoundaryKind;std::memcpy(copy.riemannBoundaryKind,s->riemannBoundaryKind,1024*sizeof(*s->riemannBoundaryKind));
copy.riemannBoundaryRhoFix=reference->riemannBoundaryRhoFix;std::memcpy(copy.riemannBoundaryRhoFix,s->riemannBoundaryRhoFix,1024*sizeof(*s->riemannBoundaryRhoFix));
copy.riemannBoundaryUFix=reference->riemannBoundaryUFix;std::memcpy(copy.riemannBoundaryUFix,s->riemannBoundaryUFix,1024*sizeof(*s->riemannBoundaryUFix));
copy.riemannBoundaryPFix=reference->riemannBoundaryPFix;std::memcpy(copy.riemannBoundaryPFix,s->riemannBoundaryPFix,1024*sizeof(*s->riemannBoundaryPFix));
copy.riemannBoundaryTFix=reference->riemannBoundaryTFix;std::memcpy(copy.riemannBoundaryTFix,s->riemannBoundaryTFix,1024*sizeof(*s->riemannBoundaryTFix));
copy.riemannBoundaryPWave=reference->riemannBoundaryPWave;std::memcpy(copy.riemannBoundaryPWave,s->riemannBoundaryPWave,1024*sizeof(*s->riemannBoundaryPWave));
copy.riemannBoundaryPWaveGamma=reference->riemannBoundaryPWaveGamma;std::memcpy(copy.riemannBoundaryPWaveGamma,s->riemannBoundaryPWaveGamma,1024*sizeof(*s->riemannBoundaryPWaveGamma));
copy.riemannBoundaryPWaveFieldInf=reference->riemannBoundaryPWaveFieldInf;std::memcpy(copy.riemannBoundaryPWaveFieldInf,s->riemannBoundaryPWaveFieldInf,1024*sizeof(*s->riemannBoundaryPWaveFieldInf));
copy.riemannBoundaryPWaveLInf=reference->riemannBoundaryPWaveLInf;std::memcpy(copy.riemannBoundaryPWaveLInf,s->riemannBoundaryPWaveLInf,1024*sizeof(*s->riemannBoundaryPWaveLInf));
copy.riemannBoundaryRho=reference->riemannBoundaryRho;std::memcpy(copy.riemannBoundaryRho,s->riemannBoundaryRho,1024*sizeof(*s->riemannBoundaryRho));
copy.riemannBoundaryUx=reference->riemannBoundaryUx;std::memcpy(copy.riemannBoundaryUx,s->riemannBoundaryUx,1024*sizeof(*s->riemannBoundaryUx));
copy.riemannBoundaryUy=reference->riemannBoundaryUy;std::memcpy(copy.riemannBoundaryUy,s->riemannBoundaryUy,1024*sizeof(*s->riemannBoundaryUy));
copy.riemannBoundaryUz=reference->riemannBoundaryUz;std::memcpy(copy.riemannBoundaryUz,s->riemannBoundaryUz,1024*sizeof(*s->riemannBoundaryUz));
copy.riemannBoundaryP=reference->riemannBoundaryP;std::memcpy(copy.riemannBoundaryP,s->riemannBoundaryP,1024*sizeof(*s->riemannBoundaryP));
copy.riemannBoundaryT=reference->riemannBoundaryT;std::memcpy(copy.riemannBoundaryT,s->riemannBoundaryT,1024*sizeof(*s->riemannBoundaryT));
copy.scheduledInletFaceMask=reference->scheduledInletFaceMask;std::memcpy(copy.scheduledInletFaceMask,s->scheduledInletFaceMask,1024*sizeof(*s->scheduledInletFaceMask));
copy.pressureScheduleTimes=reference->pressureScheduleTimes;std::memcpy(copy.pressureScheduleTimes,s->pressureScheduleTimes,1024*sizeof(*s->pressureScheduleTimes));
copy.pressureScheduleValues=reference->pressureScheduleValues;std::memcpy(copy.pressureScheduleValues,s->pressureScheduleValues,1024*sizeof(*s->pressureScheduleValues));
copy.volumeFractionScheduleTimes=reference->volumeFractionScheduleTimes;std::memcpy(copy.volumeFractionScheduleTimes,s->volumeFractionScheduleTimes,1024*sizeof(*s->volumeFractionScheduleTimes));
copy.volumeFractionScheduleValues=reference->volumeFractionScheduleValues;std::memcpy(copy.volumeFractionScheduleValues,s->volumeFractionScheduleValues,1024*sizeof(*s->volumeFractionScheduleValues));
copy.gradPx=reference->gradPx;std::memcpy(copy.gradPx,s->gradPx,1024*sizeof(*s->gradPx));
copy.gradPy=reference->gradPy;std::memcpy(copy.gradPy,s->gradPy,1024*sizeof(*s->gradPy));
copy.gradPz=reference->gradPz;std::memcpy(copy.gradPz,s->gradPz,1024*sizeof(*s->gradPz));
copy.gradRhoX=reference->gradRhoX;std::memcpy(copy.gradRhoX,s->gradRhoX,1024*sizeof(*s->gradRhoX));
copy.gradRhoY=reference->gradRhoY;std::memcpy(copy.gradRhoY,s->gradRhoY,1024*sizeof(*s->gradRhoY));
copy.gradRhoZ=reference->gradRhoZ;std::memcpy(copy.gradRhoZ,s->gradRhoZ,1024*sizeof(*s->gradRhoZ));
copy.gradUxX=reference->gradUxX;std::memcpy(copy.gradUxX,s->gradUxX,1024*sizeof(*s->gradUxX));
copy.gradUxY=reference->gradUxY;std::memcpy(copy.gradUxY,s->gradUxY,1024*sizeof(*s->gradUxY));
copy.gradUxZ=reference->gradUxZ;std::memcpy(copy.gradUxZ,s->gradUxZ,1024*sizeof(*s->gradUxZ));
copy.gradUyX=reference->gradUyX;std::memcpy(copy.gradUyX,s->gradUyX,1024*sizeof(*s->gradUyX));
copy.gradUyY=reference->gradUyY;std::memcpy(copy.gradUyY,s->gradUyY,1024*sizeof(*s->gradUyY));
copy.gradUyZ=reference->gradUyZ;std::memcpy(copy.gradUyZ,s->gradUyZ,1024*sizeof(*s->gradUyZ));
copy.gradUzX=reference->gradUzX;std::memcpy(copy.gradUzX,s->gradUzX,1024*sizeof(*s->gradUzX));
copy.gradUzY=reference->gradUzY;std::memcpy(copy.gradUzY,s->gradUzY,1024*sizeof(*s->gradUzY));
copy.gradUzZ=reference->gradUzZ;std::memcpy(copy.gradUzZ,s->gradUzZ,1024*sizeof(*s->gradUzZ));
copy.gradTX=reference->gradTX;std::memcpy(copy.gradTX,s->gradTX,1024*sizeof(*s->gradTX));
copy.gradTY=reference->gradTY;std::memcpy(copy.gradTY,s->gradTY,1024*sizeof(*s->gradTY));
copy.gradTZ=reference->gradTZ;std::memcpy(copy.gradTZ,s->gradTZ,1024*sizeof(*s->gradTZ));
copy.gasGradientLimiterRho=reference->gasGradientLimiterRho;std::memcpy(copy.gasGradientLimiterRho,s->gasGradientLimiterRho,1024*sizeof(*s->gasGradientLimiterRho));
copy.gasGradientLimiterUx=reference->gasGradientLimiterUx;std::memcpy(copy.gasGradientLimiterUx,s->gasGradientLimiterUx,1024*sizeof(*s->gasGradientLimiterUx));
copy.gasGradientLimiterUy=reference->gasGradientLimiterUy;std::memcpy(copy.gasGradientLimiterUy,s->gasGradientLimiterUy,1024*sizeof(*s->gasGradientLimiterUy));
copy.gasGradientLimiterUz=reference->gasGradientLimiterUz;std::memcpy(copy.gasGradientLimiterUz,s->gasGradientLimiterUz,1024*sizeof(*s->gasGradientLimiterUz));
copy.gasGradientLimiterP=reference->gasGradientLimiterP;std::memcpy(copy.gasGradientLimiterP,s->gasGradientLimiterP,1024*sizeof(*s->gasGradientLimiterP));
copy.gasGradientLimiterT=reference->gasGradientLimiterT;std::memcpy(copy.gasGradientLimiterT,s->gasGradientLimiterT,1024*sizeof(*s->gasGradientLimiterT));
copy.nut=reference->nut;std::memcpy(copy.nut,s->nut,1024*sizeof(*s->nut));
copy.gasDiffusionNumber=reference->gasDiffusionNumber;std::memcpy(copy.gasDiffusionNumber,s->gasDiffusionNumber,1024*sizeof(*s->gasDiffusionNumber));
copy.rhoK=reference->rhoK;std::memcpy(copy.rhoK,s->rhoK,1024*sizeof(*s->rhoK));
copy.rhoOmega=reference->rhoOmega;std::memcpy(copy.rhoOmega,s->rhoOmega,1024*sizeof(*s->rhoOmega));
copy.rhoKInitial=reference->rhoKInitial;std::memcpy(copy.rhoKInitial,s->rhoKInitial,1024*sizeof(*s->rhoKInitial));
copy.rhoOmegaInitial=reference->rhoOmegaInitial;std::memcpy(copy.rhoOmegaInitial,s->rhoOmegaInitial,1024*sizeof(*s->rhoOmegaInitial));
copy.k=reference->k;std::memcpy(copy.k,s->k,1024*sizeof(*s->k));
copy.omega=reference->omega;std::memcpy(copy.omega,s->omega,1024*sizeof(*s->omega));
copy.sstWallDistance=reference->sstWallDistance;std::memcpy(copy.sstWallDistance,s->sstWallDistance,1024*sizeof(*s->sstWallDistance));
copy.sstF1=reference->sstF1;std::memcpy(copy.sstF1,s->sstF1,1024*sizeof(*s->sstF1));
copy.sstF2=reference->sstF2;std::memcpy(copy.sstF2,s->sstF2,1024*sizeof(*s->sstF2));
copy.gradKX=reference->gradKX;std::memcpy(copy.gradKX,s->gradKX,1024*sizeof(*s->gradKX));
copy.gradKY=reference->gradKY;std::memcpy(copy.gradKY,s->gradKY,1024*sizeof(*s->gradKY));
copy.gradKZ=reference->gradKZ;std::memcpy(copy.gradKZ,s->gradKZ,1024*sizeof(*s->gradKZ));
copy.gradOmegaX=reference->gradOmegaX;std::memcpy(copy.gradOmegaX,s->gradOmegaX,1024*sizeof(*s->gradOmegaX));
copy.gradOmegaY=reference->gradOmegaY;std::memcpy(copy.gradOmegaY,s->gradOmegaY,1024*sizeof(*s->gradOmegaY));
copy.gradOmegaZ=reference->gradOmegaZ;std::memcpy(copy.gradOmegaZ,s->gradOmegaZ,1024*sizeof(*s->gradOmegaZ));
copy.sstPhiRhoK=reference->sstPhiRhoK;std::memcpy(copy.sstPhiRhoK,s->sstPhiRhoK,1024*sizeof(*s->sstPhiRhoK));
copy.sstPhiRhoOmega=reference->sstPhiRhoOmega;std::memcpy(copy.sstPhiRhoOmega,s->sstPhiRhoOmega,1024*sizeof(*s->sstPhiRhoOmega));
copy.sstBoundaryKMode=reference->sstBoundaryKMode;std::memcpy(copy.sstBoundaryKMode,s->sstBoundaryKMode,1024*sizeof(*s->sstBoundaryKMode));
copy.sstBoundaryOmegaMode=reference->sstBoundaryOmegaMode;std::memcpy(copy.sstBoundaryOmegaMode,s->sstBoundaryOmegaMode,1024*sizeof(*s->sstBoundaryOmegaMode));
copy.sstBoundaryK=reference->sstBoundaryK;std::memcpy(copy.sstBoundaryK,s->sstBoundaryK,1024*sizeof(*s->sstBoundaryK));
copy.sstBoundaryOmega=reference->sstBoundaryOmega;std::memcpy(copy.sstBoundaryOmega,s->sstBoundaryOmega,1024*sizeof(*s->sstBoundaryOmega));
copy.sstSourceNumber=reference->sstSourceNumber;std::memcpy(copy.sstSourceNumber,s->sstSourceNumber,1024*sizeof(*s->sstSourceNumber));
copy.Ux=reference->Ux;std::memcpy(copy.Ux,s->Ux,1024*sizeof(*s->Ux));
copy.Uy=reference->Uy;std::memcpy(copy.Uy,s->Uy,1024*sizeof(*s->Uy));
copy.Uz=reference->Uz;std::memcpy(copy.Uz,s->Uz,1024*sizeof(*s->Uz));
copy.p=reference->p;std::memcpy(copy.p,s->p,1024*sizeof(*s->p));
copy.Tgas=reference->Tgas;std::memcpy(copy.Tgas,s->Tgas,1024*sizeof(*s->Tgas));
copy.couplingRhoOld=reference->couplingRhoOld;std::memcpy(copy.couplingRhoOld,s->couplingRhoOld,1024*sizeof(*s->couplingRhoOld));
copy.couplingUxOld=reference->couplingUxOld;std::memcpy(copy.couplingUxOld,s->couplingUxOld,1024*sizeof(*s->couplingUxOld));
copy.couplingUyOld=reference->couplingUyOld;std::memcpy(copy.couplingUyOld,s->couplingUyOld,1024*sizeof(*s->couplingUyOld));
copy.couplingUzOld=reference->couplingUzOld;std::memcpy(copy.couplingUzOld,s->couplingUzOld,1024*sizeof(*s->couplingUzOld));
copy.couplingTgasOld=reference->couplingTgasOld;std::memcpy(copy.couplingTgasOld,s->couplingTgasOld,1024*sizeof(*s->couplingTgasOld));
copy.particleHeatCoefficient=reference->particleHeatCoefficient;std::memcpy(copy.particleHeatCoefficient,s->particleHeatCoefficient,1024*sizeof(*s->particleHeatCoefficient));
copy.particleHeatBudget=reference->particleHeatBudget;std::memcpy(copy.particleHeatBudget,s->particleHeatBudget,1024*sizeof(*s->particleHeatBudget));
copy.epsS=reference->epsS;std::memcpy(copy.epsS,s->epsS,1024*sizeof(*s->epsS));
copy.rhoUsx=reference->rhoUsx;std::memcpy(copy.rhoUsx,s->rhoUsx,1024*sizeof(*s->rhoUsx));
copy.rhoUsy=reference->rhoUsy;std::memcpy(copy.rhoUsy,s->rhoUsy,1024*sizeof(*s->rhoUsy));
copy.rhoUsz=reference->rhoUsz;std::memcpy(copy.rhoUsz,s->rhoUsz,1024*sizeof(*s->rhoUsz));
copy.rhoEs=reference->rhoEs;std::memcpy(copy.rhoEs,s->rhoEs,1024*sizeof(*s->rhoEs));
copy.rhoDs=reference->rhoDs;std::memcpy(copy.rhoDs,s->rhoDs,1024*sizeof(*s->rhoDs));
copy.rhoHp=reference->rhoHp;std::memcpy(copy.rhoHp,s->rhoHp,1024*sizeof(*s->rhoHp));
copy.Usx=reference->Usx;std::memcpy(copy.Usx,s->Usx,1024*sizeof(*s->Usx));
copy.Usy=reference->Usy;std::memcpy(copy.Usy,s->Usy,1024*sizeof(*s->Usy));
copy.Usz=reference->Usz;std::memcpy(copy.Usz,s->Usz,1024*sizeof(*s->Usz));
copy.theta=reference->theta;std::memcpy(copy.theta,s->theta,1024*sizeof(*s->theta));
copy.Tp=reference->Tp;std::memcpy(copy.Tp,s->Tp,1024*sizeof(*s->Tp));
copy.dMeanCell=reference->dMeanCell;std::memcpy(copy.dMeanCell,s->dMeanCell,1024*sizeof(*s->dMeanCell));
copy.epsGPrev=reference->epsGPrev;std::memcpy(copy.epsGPrev,s->epsGPrev,1024*sizeof(*s->epsGPrev));
copy.collisionalPressure=reference->collisionalPressure;std::memcpy(copy.collisionalPressure,s->collisionalPressure,1024*sizeof(*s->collisionalPressure));
copy.pressureKickScale=reference->pressureKickScale;std::memcpy(copy.pressureKickScale,s->pressureKickScale,1024*sizeof(*s->pressureKickScale));
copy.pressureDeltaMomX=reference->pressureDeltaMomX;std::memcpy(copy.pressureDeltaMomX,s->pressureDeltaMomX,1024*sizeof(*s->pressureDeltaMomX));
copy.pressureDeltaMomY=reference->pressureDeltaMomY;std::memcpy(copy.pressureDeltaMomY,s->pressureDeltaMomY,1024*sizeof(*s->pressureDeltaMomY));
copy.pressureDeltaMomZ=reference->pressureDeltaMomZ;std::memcpy(copy.pressureDeltaMomZ,s->pressureDeltaMomZ,1024*sizeof(*s->pressureDeltaMomZ));
copy.pressureDeltaEnergy=reference->pressureDeltaEnergy;std::memcpy(copy.pressureDeltaEnergy,s->pressureDeltaEnergy,1024*sizeof(*s->pressureDeltaEnergy));
copy.pressureParticleMoments=reference->pressureParticleMoments;std::memcpy(copy.pressureParticleMoments,s->pressureParticleMoments,1024*sizeof(*s->pressureParticleMoments));
copy.pressureParticleCount=reference->pressureParticleCount;std::memcpy(copy.pressureParticleCount,s->pressureParticleCount,1024*sizeof(*s->pressureParticleCount));
copy.solidPressurePhiMomX=reference->solidPressurePhiMomX;std::memcpy(copy.solidPressurePhiMomX,s->solidPressurePhiMomX,1024*sizeof(*s->solidPressurePhiMomX));
copy.solidPressurePhiMomY=reference->solidPressurePhiMomY;std::memcpy(copy.solidPressurePhiMomY,s->solidPressurePhiMomY,1024*sizeof(*s->solidPressurePhiMomY));
copy.solidPressurePhiMomZ=reference->solidPressurePhiMomZ;std::memcpy(copy.solidPressurePhiMomZ,s->solidPressurePhiMomZ,1024*sizeof(*s->solidPressurePhiMomZ));
copy.solidPressurePhiEnergy=reference->solidPressurePhiEnergy;std::memcpy(copy.solidPressurePhiEnergy,s->solidPressurePhiEnergy,1024*sizeof(*s->solidPressurePhiEnergy));
copy.mobilePackingRho=reference->mobilePackingRho;std::memcpy(copy.mobilePackingRho,s->mobilePackingRho,1024*sizeof(*s->mobilePackingRho));
copy.packingStuckRho=reference->packingStuckRho;std::memcpy(copy.packingStuckRho,s->packingStuckRho,1024*sizeof(*s->packingStuckRho));
copy.mobilePackingMomX=reference->mobilePackingMomX;std::memcpy(copy.mobilePackingMomX,s->mobilePackingMomX,1024*sizeof(*s->mobilePackingMomX));
copy.mobilePackingMomY=reference->mobilePackingMomY;std::memcpy(copy.mobilePackingMomY,s->mobilePackingMomY,1024*sizeof(*s->mobilePackingMomY));
copy.mobilePackingMomZ=reference->mobilePackingMomZ;std::memcpy(copy.mobilePackingMomZ,s->mobilePackingMomZ,1024*sizeof(*s->mobilePackingMomZ));
copy.mobilePackingActiveCellMask=reference->mobilePackingActiveCellMask;std::memcpy(copy.mobilePackingActiveCellMask,s->mobilePackingActiveCellMask,1024*sizeof(*s->mobilePackingActiveCellMask));
copy.mobilePackingCorrectionCellMask=reference->mobilePackingCorrectionCellMask;std::memcpy(copy.mobilePackingCorrectionCellMask,s->mobilePackingCorrectionCellMask,1024*sizeof(*s->mobilePackingCorrectionCellMask));
copy.mobilePackingActiveCellList=reference->mobilePackingActiveCellList;std::memcpy(copy.mobilePackingActiveCellList,s->mobilePackingActiveCellList,1024*sizeof(*s->mobilePackingActiveCellList));
copy.mobilePackingCorrectionCellList=reference->mobilePackingCorrectionCellList;std::memcpy(copy.mobilePackingCorrectionCellList,s->mobilePackingCorrectionCellList,1024*sizeof(*s->mobilePackingCorrectionCellList));
copy.mobilePackingFrontierCurrent=reference->mobilePackingFrontierCurrent;std::memcpy(copy.mobilePackingFrontierCurrent,s->mobilePackingFrontierCurrent,1024*sizeof(*s->mobilePackingFrontierCurrent));
copy.mobilePackingFrontierNext=reference->mobilePackingFrontierNext;std::memcpy(copy.mobilePackingFrontierNext,s->mobilePackingFrontierNext,1024*sizeof(*s->mobilePackingFrontierNext));
copy.mobilePackingActiveCellCount=reference->mobilePackingActiveCellCount;std::memcpy(copy.mobilePackingActiveCellCount,s->mobilePackingActiveCellCount,1024*sizeof(*s->mobilePackingActiveCellCount));
copy.mobilePackingCorrectionCellCount=reference->mobilePackingCorrectionCellCount;std::memcpy(copy.mobilePackingCorrectionCellCount,s->mobilePackingCorrectionCellCount,1024*sizeof(*s->mobilePackingCorrectionCellCount));
copy.mobilePackingFrontierCurrentCount=reference->mobilePackingFrontierCurrentCount;std::memcpy(copy.mobilePackingFrontierCurrentCount,s->mobilePackingFrontierCurrentCount,1024*sizeof(*s->mobilePackingFrontierCurrentCount));
copy.mobilePackingFrontierNextCount=reference->mobilePackingFrontierNextCount;std::memcpy(copy.mobilePackingFrontierNextCount,s->mobilePackingFrontierNextCount,1024*sizeof(*s->mobilePackingFrontierNextCount));
copy.thetaDragAlpha=reference->thetaDragAlpha;std::memcpy(copy.thetaDragAlpha,s->thetaDragAlpha,1024*sizeof(*s->thetaDragAlpha));
copy.momRhoP=reference->momRhoP;std::memcpy(copy.momRhoP,s->momRhoP,1024*sizeof(*s->momRhoP));
copy.momRhoUPx=reference->momRhoUPx;std::memcpy(copy.momRhoUPx,s->momRhoUPx,1024*sizeof(*s->momRhoUPx));
copy.momRhoUPy=reference->momRhoUPy;std::memcpy(copy.momRhoUPy,s->momRhoUPy,1024*sizeof(*s->momRhoUPy));
copy.momRhoUPz=reference->momRhoUPz;std::memcpy(copy.momRhoUPz,s->momRhoUPz,1024*sizeof(*s->momRhoUPz));
copy.momRhoEP=reference->momRhoEP;std::memcpy(copy.momRhoEP,s->momRhoEP,1024*sizeof(*s->momRhoEP));
copy.momRhoPD=reference->momRhoPD;std::memcpy(copy.momRhoPD,s->momRhoPD,1024*sizeof(*s->momRhoPD));
copy.momRhoHpP=reference->momRhoHpP;std::memcpy(copy.momRhoHpP,s->momRhoHpP,1024*sizeof(*s->momRhoHpP));
copy.radiationMobileMass=reference->radiationMobileMass;std::memcpy(copy.radiationMobileMass,s->radiationMobileMass,1024*sizeof(*s->radiationMobileMass));
copy.radiationMobileTemperatureMass=reference->radiationMobileTemperatureMass;std::memcpy(copy.radiationMobileTemperatureMass,s->radiationMobileTemperatureMass,1024*sizeof(*s->radiationMobileTemperatureMass));
copy.radiationMobileDiameterMass=reference->radiationMobileDiameterMass;std::memcpy(copy.radiationMobileDiameterMass,s->radiationMobileDiameterMass,1024*sizeof(*s->radiationMobileDiameterMass));
copy.poolThermalCount=reference->poolThermalCount;std::memcpy(copy.poolThermalCount,s->poolThermalCount,1024*sizeof(*s->poolThermalCount));
copy.poolThermalSumUx=reference->poolThermalSumUx;std::memcpy(copy.poolThermalSumUx,s->poolThermalSumUx,1024*sizeof(*s->poolThermalSumUx));
copy.poolThermalSumUy=reference->poolThermalSumUy;std::memcpy(copy.poolThermalSumUy,s->poolThermalSumUy,1024*sizeof(*s->poolThermalSumUy));
copy.poolThermalSumUz=reference->poolThermalSumUz;std::memcpy(copy.poolThermalSumUz,s->poolThermalSumUz,1024*sizeof(*s->poolThermalSumUz));
copy.poolThermalSumU2=reference->poolThermalSumU2;std::memcpy(copy.poolThermalSumU2,s->poolThermalSumU2,1024*sizeof(*s->poolThermalSumU2));
copy.poissonPoolSampleTargetCount=reference->poissonPoolSampleTargetCount;std::memcpy(copy.poissonPoolSampleTargetCount,s->poissonPoolSampleTargetCount,1024*sizeof(*s->poissonPoolSampleTargetCount));
copy.poissonPoolMass=reference->poissonPoolMass;std::memcpy(copy.poissonPoolMass,s->poissonPoolMass,1024*sizeof(*s->poissonPoolMass));
copy.poissonPoolMomX=reference->poissonPoolMomX;std::memcpy(copy.poissonPoolMomX,s->poissonPoolMomX,1024*sizeof(*s->poissonPoolMomX));
copy.poissonPoolMomY=reference->poissonPoolMomY;std::memcpy(copy.poissonPoolMomY,s->poissonPoolMomY,1024*sizeof(*s->poissonPoolMomY));
copy.poissonPoolMomZ=reference->poissonPoolMomZ;std::memcpy(copy.poissonPoolMomZ,s->poissonPoolMomZ,1024*sizeof(*s->poissonPoolMomZ));
copy.poissonPoolEnergy=reference->poissonPoolEnergy;std::memcpy(copy.poissonPoolEnergy,s->poissonPoolEnergy,1024*sizeof(*s->poissonPoolEnergy));
copy.poissonPoolDiameter=reference->poissonPoolDiameter;std::memcpy(copy.poissonPoolDiameter,s->poissonPoolDiameter,1024*sizeof(*s->poissonPoolDiameter));
copy.poissonPoolDiameter2=reference->poissonPoolDiameter2;std::memcpy(copy.poissonPoolDiameter2,s->poissonPoolDiameter2,1024*sizeof(*s->poissonPoolDiameter2));
copy.px=reference->px;std::memcpy(copy.px,s->px,1024*sizeof(*s->px));
copy.py=reference->py;std::memcpy(copy.py,s->py,1024*sizeof(*s->py));
copy.pz=reference->pz;std::memcpy(copy.pz,s->pz,1024*sizeof(*s->pz));
copy.pux=reference->pux;std::memcpy(copy.pux,s->pux,1024*sizeof(*s->pux));
copy.puy=reference->puy;std::memcpy(copy.puy,s->puy,1024*sizeof(*s->puy));
copy.puz=reference->puz;std::memcpy(copy.puz,s->puz,1024*sizeof(*s->puz));
copy.puxOld=reference->puxOld;std::memcpy(copy.puxOld,s->puxOld,1024*sizeof(*s->puxOld));
copy.puyOld=reference->puyOld;std::memcpy(copy.puyOld,s->puyOld,1024*sizeof(*s->puyOld));
copy.puzOld=reference->puzOld;std::memcpy(copy.puzOld,s->puzOld,1024*sizeof(*s->puzOld));
copy.pT=reference->pT;std::memcpy(copy.pT,s->pT,1024*sizeof(*s->pT));
copy.pThermalEnthalpyResidual=reference->pThermalEnthalpyResidual;std::memcpy(copy.pThermalEnthalpyResidual,s->pThermalEnthalpyResidual,1024*sizeof(*s->pThermalEnthalpyResidual));
copy.pTheta=reference->pTheta;std::memcpy(copy.pTheta,s->pTheta,1024*sizeof(*s->pTheta));
copy.pContactAge=reference->pContactAge;std::memcpy(copy.pContactAge,s->pContactAge,1024*sizeof(*s->pContactAge));
copy.pd=reference->pd;std::memcpy(copy.pd,s->pd,1024*sizeof(*s->pd));
copy.pm=reference->pm;std::memcpy(copy.pm,s->pm,1024*sizeof(*s->pm));
copy.pCellId=reference->pCellId;std::memcpy(copy.pCellId,s->pCellId,1024*sizeof(*s->pCellId));
copy.pStatus=reference->pStatus;std::memcpy(copy.pStatus,s->pStatus,1024*sizeof(*s->pStatus));
copy.pStuck=reference->pStuck;std::memcpy(copy.pStuck,s->pStuck,1024*sizeof(*s->pStuck));
copy.pStuckFaceId=reference->pStuckFaceId;std::memcpy(copy.pStuckFaceId,s->pStuckFaceId,1024*sizeof(*s->pStuckFaceId));
copy.pDepositionArea=reference->pDepositionArea;std::memcpy(copy.pDepositionArea,s->pDepositionArea,1024*sizeof(*s->pDepositionArea));
copy.pContactDuration=reference->pContactDuration;std::memcpy(copy.pContactDuration,s->pContactDuration,1024*sizeof(*s->pContactDuration));
copy.pContactMaximumArea=reference->pContactMaximumArea;std::memcpy(copy.pContactMaximumArea,s->pContactMaximumArea,1024*sizeof(*s->pContactMaximumArea));
copy.pContactPeakFraction=reference->pContactPeakFraction;std::memcpy(copy.pContactPeakFraction,s->pContactPeakFraction,1024*sizeof(*s->pContactPeakFraction));
copy.pColdNodeSpecificEnthalpy=reference->pColdNodeSpecificEnthalpy;std::memcpy(copy.pColdNodeSpecificEnthalpy,s->pColdNodeSpecificEnthalpy,1024*sizeof(*s->pColdNodeSpecificEnthalpy));
copy.pColdRingSolidMass=reference->pColdRingSolidMass;std::memcpy(copy.pColdRingSolidMass,s->pColdRingSolidMass,1024*sizeof(*s->pColdRingSolidMass));
copy.pColdFrozenArea=reference->pColdFrozenArea;std::memcpy(copy.pColdFrozenArea,s->pColdFrozenArea,1024*sizeof(*s->pColdFrozenArea));
copy.pColdContactAge=reference->pColdContactAge;std::memcpy(copy.pColdContactAge,s->pColdContactAge,1024*sizeof(*s->pColdContactAge));
copy.pCold2DNodeSpecificEnthalpy=reference->pCold2DNodeSpecificEnthalpy;std::memcpy(copy.pCold2DNodeSpecificEnthalpy,s->pCold2DNodeSpecificEnthalpy,1024*sizeof(*s->pCold2DNodeSpecificEnthalpy));
copy.pCold2DRingContactAge=reference->pCold2DRingContactAge;std::memcpy(copy.pCold2DRingContactAge,s->pCold2DRingContactAge,1024*sizeof(*s->pCold2DRingContactAge));
copy.pCold2DFrozenArea=reference->pCold2DFrozenArea;std::memcpy(copy.pCold2DFrozenArea,s->pCold2DFrozenArea,1024*sizeof(*s->pCold2DFrozenArea));
copy.pRng=reference->pRng;std::memcpy(copy.pRng,s->pRng,1024*sizeof(*s->pRng));
copy.pOrigId=reference->pOrigId;std::memcpy(copy.pOrigId,s->pOrigId,1024*sizeof(*s->pOrigId));
copy.wallBoundParticleIndex=reference->wallBoundParticleIndex;std::memcpy(copy.wallBoundParticleIndex,s->wallBoundParticleIndex,1024*sizeof(*s->wallBoundParticleIndex));
copy.wallBoundParticleCountDevice=reference->wallBoundParticleCountDevice;std::memcpy(copy.wallBoundParticleCountDevice,s->wallBoundParticleCountDevice,1024*sizeof(*s->wallBoundParticleCountDevice));
copy.sortedParticleIndex=reference->sortedParticleIndex;std::memcpy(copy.sortedParticleIndex,s->sortedParticleIndex,1024*sizeof(*s->sortedParticleIndex));
copy.cellParticleCount=reference->cellParticleCount;std::memcpy(copy.cellParticleCount,s->cellParticleCount,1024*sizeof(*s->cellParticleCount));
copy.cellParticleOffset=reference->cellParticleOffset;std::memcpy(copy.cellParticleOffset,s->cellParticleOffset,1024*sizeof(*s->cellParticleOffset));
copy.compactCellOffset=reference->compactCellOffset;std::memcpy(copy.compactCellOffset,s->compactCellOffset,1024*sizeof(*s->compactCellOffset));
copy.cellParticleWrite=reference->cellParticleWrite;std::memcpy(copy.cellParticleWrite,s->cellParticleWrite,1024*sizeof(*s->cellParticleWrite));
copy.preBaseCellOffset=reference->preBaseCellOffset;std::memcpy(copy.preBaseCellOffset,s->preBaseCellOffset,1024*sizeof(*s->preBaseCellOffset));
copy.preBaseParticleCountDevice=reference->preBaseParticleCountDevice;std::memcpy(copy.preBaseParticleCountDevice,s->preBaseParticleCountDevice,1024*sizeof(*s->preBaseParticleCountDevice));
copy.cellScanTempStorage=reference->cellScanTempStorage;std::memcpy(copy.cellScanTempStorage,s->cellScanTempStorage,1024*sizeof(*s->cellScanTempStorage));
copy.compactCountDevice=reference->compactCountDevice;std::memcpy(copy.compactCountDevice,s->compactCountDevice,1024*sizeof(*s->compactCountDevice));
copy.compactSelectTempStorage=reference->compactSelectTempStorage;std::memcpy(copy.compactSelectTempStorage,s->compactSelectTempStorage,1024*sizeof(*s->compactSelectTempStorage));
copy.csrCellTaskCount=reference->csrCellTaskCount;std::memcpy(copy.csrCellTaskCount,s->csrCellTaskCount,1024*sizeof(*s->csrCellTaskCount));
copy.csrCellTaskOffset=reference->csrCellTaskOffset;std::memcpy(copy.csrCellTaskOffset,s->csrCellTaskOffset,1024*sizeof(*s->csrCellTaskOffset));
copy.csrMultiTaskCellList=reference->csrMultiTaskCellList;std::memcpy(copy.csrMultiTaskCellList,s->csrMultiTaskCellList,1024*sizeof(*s->csrMultiTaskCellList));
copy.csrHeavyCellList=reference->csrHeavyCellList;std::memcpy(copy.csrHeavyCellList,s->csrHeavyCellList,1024*sizeof(*s->csrHeavyCellList));
copy.csrHeavyTaskCell=reference->csrHeavyTaskCell;std::memcpy(copy.csrHeavyTaskCell,s->csrHeavyTaskCell,1024*sizeof(*s->csrHeavyTaskCell));
copy.csrHeavyTaskBegin=reference->csrHeavyTaskBegin;std::memcpy(copy.csrHeavyTaskBegin,s->csrHeavyTaskBegin,1024*sizeof(*s->csrHeavyTaskBegin));
copy.csrHeavyTaskEnd=reference->csrHeavyTaskEnd;std::memcpy(copy.csrHeavyTaskEnd,s->csrHeavyTaskEnd,1024*sizeof(*s->csrHeavyTaskEnd));
copy.csrHeavyCellTaskStart=reference->csrHeavyCellTaskStart;std::memcpy(copy.csrHeavyCellTaskStart,s->csrHeavyCellTaskStart,1024*sizeof(*s->csrHeavyCellTaskStart));
copy.csrHeavyCellTaskCount=reference->csrHeavyCellTaskCount;std::memcpy(copy.csrHeavyCellTaskCount,s->csrHeavyCellTaskCount,1024*sizeof(*s->csrHeavyCellTaskCount));
copy.csrHeavyPartials=reference->csrHeavyPartials;std::memcpy(copy.csrHeavyPartials,s->csrHeavyPartials,1024*sizeof(*s->csrHeavyPartials));
copy.csrHeavyInjectionTaskCount=reference->csrHeavyInjectionTaskCount;std::memcpy(copy.csrHeavyInjectionTaskCount,s->csrHeavyInjectionTaskCount,1024*sizeof(*s->csrHeavyInjectionTaskCount));
copy.csrHeavyInjectionTaskCursor=reference->csrHeavyInjectionTaskCursor;std::memcpy(copy.csrHeavyInjectionTaskCursor,s->csrHeavyInjectionTaskCursor,1024*sizeof(*s->csrHeavyInjectionTaskCursor));
copy.csrHeavyInjectionTaskCell=reference->csrHeavyInjectionTaskCell;std::memcpy(copy.csrHeavyInjectionTaskCell,s->csrHeavyInjectionTaskCell,1024*sizeof(*s->csrHeavyInjectionTaskCell));
copy.csrHeavyInjectionTaskBegin=reference->csrHeavyInjectionTaskBegin;std::memcpy(copy.csrHeavyInjectionTaskBegin,s->csrHeavyInjectionTaskBegin,1024*sizeof(*s->csrHeavyInjectionTaskBegin));
copy.csrHeavyInjectionTaskEnd=reference->csrHeavyInjectionTaskEnd;std::memcpy(copy.csrHeavyInjectionTaskEnd,s->csrHeavyInjectionTaskEnd,1024*sizeof(*s->csrHeavyInjectionTaskEnd));
copy.csrHeavyInjectionCellTaskStart=reference->csrHeavyInjectionCellTaskStart;std::memcpy(copy.csrHeavyInjectionCellTaskStart,s->csrHeavyInjectionCellTaskStart,1024*sizeof(*s->csrHeavyInjectionCellTaskStart));
copy.csrHeavyInjectionCellTaskCount=reference->csrHeavyInjectionCellTaskCount;std::memcpy(copy.csrHeavyInjectionCellTaskCount,s->csrHeavyInjectionCellTaskCount,1024*sizeof(*s->csrHeavyInjectionCellTaskCount));
copy.csrHeavyInjectionPartials=reference->csrHeavyInjectionPartials;std::memcpy(copy.csrHeavyInjectionPartials,s->csrHeavyInjectionPartials,1024*sizeof(*s->csrHeavyInjectionPartials));
copy.sourceCell=reference->sourceCell;std::memcpy(copy.sourceCell,s->sourceCell,1024*sizeof(*s->sourceCell));
copy.sourceFace=reference->sourceFace;std::memcpy(copy.sourceFace,s->sourceFace,1024*sizeof(*s->sourceFace));
copy.sourcePx=reference->sourcePx;std::memcpy(copy.sourcePx,s->sourcePx,1024*sizeof(*s->sourcePx));
copy.sourcePy=reference->sourcePy;std::memcpy(copy.sourcePy,s->sourcePy,1024*sizeof(*s->sourcePy));
copy.sourcePz=reference->sourcePz;std::memcpy(copy.sourcePz,s->sourcePz,1024*sizeof(*s->sourcePz));
copy.sourceUx=reference->sourceUx;std::memcpy(copy.sourceUx,s->sourceUx,1024*sizeof(*s->sourceUx));
copy.sourceUy=reference->sourceUy;std::memcpy(copy.sourceUy,s->sourceUy,1024*sizeof(*s->sourceUy));
copy.sourceUz=reference->sourceUz;std::memcpy(copy.sourceUz,s->sourceUz,1024*sizeof(*s->sourceUz));
copy.sourceT=reference->sourceT;std::memcpy(copy.sourceT,s->sourceT,1024*sizeof(*s->sourceT));
copy.sourceTheta=reference->sourceTheta;std::memcpy(copy.sourceTheta,s->sourceTheta,1024*sizeof(*s->sourceTheta));
copy.sourceD=reference->sourceD;std::memcpy(copy.sourceD,s->sourceD,1024*sizeof(*s->sourceD));
copy.sourceMassRate=reference->sourceMassRate;std::memcpy(copy.sourceMassRate,s->sourceMassRate,1024*sizeof(*s->sourceMassRate));
copy.sourceResidualMass=reference->sourceResidualMass;std::memcpy(copy.sourceResidualMass,s->sourceResidualMass,1024*sizeof(*s->sourceResidualMass));
copy.compactPx=reference->compactPx;std::memcpy(copy.compactPx,s->compactPx,1024*sizeof(*s->compactPx));
copy.compactPy=reference->compactPy;std::memcpy(copy.compactPy,s->compactPy,1024*sizeof(*s->compactPy));
copy.compactPz=reference->compactPz;std::memcpy(copy.compactPz,s->compactPz,1024*sizeof(*s->compactPz));
copy.compactPux=reference->compactPux;std::memcpy(copy.compactPux,s->compactPux,1024*sizeof(*s->compactPux));
copy.compactPuxOld=reference->compactPuxOld;std::memcpy(copy.compactPuxOld,s->compactPuxOld,1024*sizeof(*s->compactPuxOld));
copy.compactPuy=reference->compactPuy;std::memcpy(copy.compactPuy,s->compactPuy,1024*sizeof(*s->compactPuy));
copy.compactPuyOld=reference->compactPuyOld;std::memcpy(copy.compactPuyOld,s->compactPuyOld,1024*sizeof(*s->compactPuyOld));
copy.compactPuz=reference->compactPuz;std::memcpy(copy.compactPuz,s->compactPuz,1024*sizeof(*s->compactPuz));
copy.compactPuzOld=reference->compactPuzOld;std::memcpy(copy.compactPuzOld,s->compactPuzOld,1024*sizeof(*s->compactPuzOld));
copy.compactPT=reference->compactPT;std::memcpy(copy.compactPT,s->compactPT,1024*sizeof(*s->compactPT));
copy.compactPThermalEnthalpyResidual=reference->compactPThermalEnthalpyResidual;std::memcpy(copy.compactPThermalEnthalpyResidual,s->compactPThermalEnthalpyResidual,1024*sizeof(*s->compactPThermalEnthalpyResidual));
copy.compactPTheta=reference->compactPTheta;std::memcpy(copy.compactPTheta,s->compactPTheta,1024*sizeof(*s->compactPTheta));
copy.compactPContactAge=reference->compactPContactAge;std::memcpy(copy.compactPContactAge,s->compactPContactAge,1024*sizeof(*s->compactPContactAge));
copy.compactPd=reference->compactPd;std::memcpy(copy.compactPd,s->compactPd,1024*sizeof(*s->compactPd));
copy.compactPm=reference->compactPm;std::memcpy(copy.compactPm,s->compactPm,1024*sizeof(*s->compactPm));
copy.compactPCellId=reference->compactPCellId;std::memcpy(copy.compactPCellId,s->compactPCellId,1024*sizeof(*s->compactPCellId));
copy.compactPStatus=reference->compactPStatus;std::memcpy(copy.compactPStatus,s->compactPStatus,1024*sizeof(*s->compactPStatus));
copy.compactPStuck=reference->compactPStuck;std::memcpy(copy.compactPStuck,s->compactPStuck,1024*sizeof(*s->compactPStuck));
copy.compactPStuckFaceId=reference->compactPStuckFaceId;std::memcpy(copy.compactPStuckFaceId,s->compactPStuckFaceId,1024*sizeof(*s->compactPStuckFaceId));
copy.compactPDepositionArea=reference->compactPDepositionArea;std::memcpy(copy.compactPDepositionArea,s->compactPDepositionArea,1024*sizeof(*s->compactPDepositionArea));
copy.compactPContactDuration=reference->compactPContactDuration;std::memcpy(copy.compactPContactDuration,s->compactPContactDuration,1024*sizeof(*s->compactPContactDuration));
copy.compactPContactMaximumArea=reference->compactPContactMaximumArea;std::memcpy(copy.compactPContactMaximumArea,s->compactPContactMaximumArea,1024*sizeof(*s->compactPContactMaximumArea));
copy.compactPContactPeakFraction=reference->compactPContactPeakFraction;std::memcpy(copy.compactPContactPeakFraction,s->compactPContactPeakFraction,1024*sizeof(*s->compactPContactPeakFraction));
copy.compactPColdNodeSpecificEnthalpy=reference->compactPColdNodeSpecificEnthalpy;std::memcpy(copy.compactPColdNodeSpecificEnthalpy,s->compactPColdNodeSpecificEnthalpy,1024*sizeof(*s->compactPColdNodeSpecificEnthalpy));
copy.compactPColdRingSolidMass=reference->compactPColdRingSolidMass;std::memcpy(copy.compactPColdRingSolidMass,s->compactPColdRingSolidMass,1024*sizeof(*s->compactPColdRingSolidMass));
copy.compactPColdFrozenArea=reference->compactPColdFrozenArea;std::memcpy(copy.compactPColdFrozenArea,s->compactPColdFrozenArea,1024*sizeof(*s->compactPColdFrozenArea));
copy.compactPColdContactAge=reference->compactPColdContactAge;std::memcpy(copy.compactPColdContactAge,s->compactPColdContactAge,1024*sizeof(*s->compactPColdContactAge));
copy.compactPCold2DNodeSpecificEnthalpy=reference->compactPCold2DNodeSpecificEnthalpy;std::memcpy(copy.compactPCold2DNodeSpecificEnthalpy,s->compactPCold2DNodeSpecificEnthalpy,1024*sizeof(*s->compactPCold2DNodeSpecificEnthalpy));
copy.compactPCold2DRingContactAge=reference->compactPCold2DRingContactAge;std::memcpy(copy.compactPCold2DRingContactAge,s->compactPCold2DRingContactAge,1024*sizeof(*s->compactPCold2DRingContactAge));
copy.compactPCold2DFrozenArea=reference->compactPCold2DFrozenArea;std::memcpy(copy.compactPCold2DFrozenArea,s->compactPCold2DFrozenArea,1024*sizeof(*s->compactPCold2DFrozenArea));
copy.compactPRng=reference->compactPRng;std::memcpy(copy.compactPRng,s->compactPRng,1024*sizeof(*s->compactPRng));
copy.compactPOrigId=reference->compactPOrigId;std::memcpy(copy.compactPOrigId,s->compactPOrigId,1024*sizeof(*s->compactPOrigId));
*reference=copy;}

applyCollisionalPressureProjectionParticlesAtomicKernel<true><<<2,32>>>(s);baselinePressureParticles<true><<<2,32>>>(reference);deviceSync();
for(int i=0;i<64;i++){
 equal(s->pux[i],reference->pux[i],"unsorted constrained particle X");equal(s->puy[i],reference->puy[i],"unsorted constrained particle Y");equal(s->puz[i],reference->puz[i],"unsorted constrained particle Z");equal(s->pTheta[i],reference->pTheta[i],"unsorted constrained theta");
 equal(s->puxOld[i],reference->puxOld[i],"constraint old velocity");
 if(s->pStuck[i])ck(s->pux[i]==0&&s->puy[i]==0&&s->puz[i]==0,"wall constraint velocity");
 if(s->pStuck[i]==Foam::gpuThermal::particleWallDeposited)ck(s->puxOld[i]==0,"deposited old velocity reset");
}
for(int j=0;j<7*16;j++){double a=s->pressureParticleMoments[j],b=reference->pressureParticleMoments[j];ck(std::isfinite(a)&&fabs(a-b)<=2e-12*fmax(1.,fabs(b)),"atomic constrained moments");}
puts("PASS bitwise unsorted particle update with mobile, transient and deposited wall constraints");
puts("PASS bitwise pressure face deltas and published cell moments");
for(int block:{32,64,128,256}){for(int f=0;f<3;f++)s->particleWallReflectedEnergy[f]=s->particleWallDepositedEnergy[f]=reference->particleWallReflectedEnergy[f]=reference->particleWallDepositedEnergy[f]=0;wallChannels<<<(129+block-1)/block,block>>>(s,reference);deviceSync();for(int f=0;f<3;f++){equal(s->particleWallReflectedEnergy[f],reference->particleWallReflectedEnergy[f],"wall reflected");equal(s->particleWallDepositedEnergy[f],reference->particleWallDepositedEnergy[f],"wall deposited");}}puts("PASS bitwise signed wall ledger with partial warps, multiple faces/channels and block sizes");
if(!std::getenv("PR14_CORE_ONLY")){
s->coldWallSolidificationEnabled=reference->coldWallSolidificationEnabled=1;
s->coldWallSolidificationParameters=reference->coldWallSolidificationParameters={2327.,20.,1.16e6,3990.,1273.,5.9,.25,0.,1,4};
// PCR/Thomas arithmetic is double; only the CHT node cache remains float.
// Use each actual profile owner for the explicit storage rounding budget.
const double coldArithmeticTolerance=128*GPU_REAL_EPSILON;
double largestColdResidual=0,largestColdBound=0,largestColdNodeError=0,largestColdWallError=0;
for(int closed:{0,1})for(int stable=0;stable<1;stable++)for(int transient:{0,1})for(int initialTemperature:{1000,2328,3500}){
 s->coldWallSolidificationParameters.wallTransientResistance=reference->coldWallSolidificationParameters.wallTransientResistance=transient;
 for(int i=0;i<8;i++){s->pColdNodeSpecificEnthalpy[i]=Foam::gpuThermal::coldWallSpecificEnthalpyJkg(Real(initialTemperature),s->coldWallSolidificationParameters);s->pColdRingSolidMass[i]=0;}
 s->pColdContactAge[0]=0;s->pColdFrozenArea[0]=0;
 for(int step=0;step<64;step++){
  seedColdReference(s,reference);
  double before=0;for(int i=0;i<8;i++)before+=double(Real(3.2e-9)/Real(8))*s->pColdNodeSpecificEnthalpy[i];
  coldScalar<<<1,1>>>(reference,values,7.5e-8,closed?0:2e-4,8e-6,3200);
  coldPair<<<1,8>>>(s,values,stable,7.5e-8,closed?0:2e-4);deviceSync();
  ck(values[0]==1&&values[1]==1&&values[8]==0&&values[9]==1,"scalar/PCR cold-wall valid step");
  nearCold(values[2],values[3],coldArithmeticTolerance*fmax(fabs(values[2]),fabs(values[3])),"cold-wall mean temperature");
  nearCold(values[4],values[5],coldArithmeticTolerance*double(Real(2.2e-8)),"cold-wall frozen area");
  double wallError=fabs(values[6]-values[7]);largestColdWallError=fmax(largestColdWallError,wallError);
  nearCold(values[6],values[7],coldArithmeticTolerance*fmax(fabs(values[6]),fabs(values[7])),"cold-wall wall energy");
  double after=0,storageBound=0;
  for(int i=0;i<8;i++){
    auto node=s->pColdNodeSpecificEnthalpy[i],refNode=reference->pColdNodeSpecificEnthalpy[i];double nodeMass=double(Real(3.2e-9)/Real(8));after+=nodeMass*node;
    double nodeError=fabs(double(node)-refNode);largestColdNodeError=fmax(largestColdNodeError,nodeError);
    nearCold(node,refNode,coldStorageUlp(node)+coldStorageUlp(refNode)+coldArithmeticTolerance*fmax(fabs(double(node)),fabs(double(refNode))),"cold-wall node enthalpy");
    nearCold(s->pColdRingSolidMass[i],reference->pColdRingSolidMass[i],coldStorageUlp(s->pColdRingSolidMass[i])+coldStorageUlp(reference->pColdRingSolidMass[i])+coldArithmeticTolerance*double(Real(3.2e-9)),"cold-wall ring mass");
    storageBound+=.5*nodeMass*coldStorageUlp(node);
  }
  nearCold(s->pColdFrozenArea[0],reference->pColdFrozenArea[0],coldStorageUlp(s->pColdFrozenArea[0])+coldStorageUlp(reference->pColdFrozenArea[0])+coldArithmeticTolerance*double(Real(2.2e-8)),"cold-wall published frozen area");
  equal(s->pColdContactAge[0],reference->pColdContactAge[0],"cold-wall time64");
  double residual=after-before+values[6]-values[10];
  double bound=storageBound+64*GPU_REAL_EPSILON*(fabs(before)+fabs(after)+fabs(values[6])+fabs(values[10]));
  largestColdResidual=fmax(largestColdResidual,fabs(residual));largestColdBound=fmax(largestColdBound,bound);
  ck(fabs(residual)<=bound,"cold-wall particle enthalpy + wall heat - gas input conservation within storage rounding");
 }
 double savedH[8],savedRing[8];for(int n=0;n<8;n++){savedH[n]=s->pColdNodeSpecificEnthalpy[n];savedRing[n]=s->pColdRingSolidMass[n];}
 double age=s->pColdContactAge[0];double frozen=s->pColdFrozenArea[0];
 for(int bad:{0,1,2,3}){
 seedColdReference(s,reference);double badDt=bad==0?-1:7.5e-8,badGas=bad==1?NAN:(bad==3?-1:0),badTemperature=bad==2?NAN:3200;
 coldScalar<<<1,1>>>(reference,values,badDt,badGas,8e-6,badTemperature);coldPair<<<1,8>>>(s,values,stable,badDt,badGas,8e-6,badTemperature);deviceSync();
 ck(values[0]==0&&values[1]==0,"invalid cold-wall step rejected");
 for(int n=0;n<8;n++){equal(s->pColdNodeSpecificEnthalpy[n],savedH[n],"rejected enthalpy unchanged");equal(s->pColdRingSolidMass[n],savedRing[n],"rejected ring mass unchanged");equal(reference->pColdNodeSpecificEnthalpy[n],savedH[n],"rejected scalar enthalpy unchanged");equal(reference->pColdRingSolidMass[n],savedRing[n],"rejected scalar ring mass unchanged");}
 equal(s->pColdContactAge[0],age,"rejected age unchanged");equal(s->pColdFrozenArea[0],frozen,"rejected frozen area unchanged");equal(reference->pColdContactAge[0],age,"rejected scalar age unchanged");equal(reference->pColdFrozenArea[0],frozen,"rejected scalar frozen area unchanged");
 }
}
// Delay wetting beyond the whole step to isolate gas relaxation. This independent
// constant-cp exponential catches both a top-node-only source and reapplying the
// gas step during every nonlinear iteration, even if both solvers share a bug.
for(int stable=0;stable<1;stable++)for(int iterations:{1,4,12})for(int gasTemperature:{700,1700}){
 auto& p=s->coldWallSolidificationParameters;p.nonlinearIterations=iterations;
 s->pColdNodeSpecificEnthalpy[0]=p.solidSpecificHeatJkgK*Real(1000);
 auto oldH=s->pColdNodeSpecificEnthalpy[0];
 for(int n=0;n<8;n++){s->pColdNodeSpecificEnthalpy[n]=oldH;s->pColdRingSolidMass[n]=0;}
 s->pColdContactAge[0]=0;s->pColdFrozenArea[0]=0;
 const double dt=2e-4,conductance=double(Real(2e-4)),cp=double(p.solidSpecificHeatJkgK),mass=double(Real(3.2e-9));
 const double expectedH=double(oldH)+(cp*gasTemperature-double(oldH))*(-std::expm1(-dt*conductance/(mass*cp)));
 coldPair<<<1,8>>>(s,values,stable,dt,conductance,1e6,gasTemperature);deviceSync();ck(values[0]==1&&values[8]==0,"gas-only cold-wall valid step");
 for(int n=0;n<8;n++)nearCold(s->pColdNodeSpecificEnthalpy[n],expectedH,.5*coldStorageUlp(s->pColdNodeSpecificEnthalpy[n])+coldArithmeticTolerance*fabs(expectedH),"uniform gas-only analytic exponential");
 nearCold(values[2],expectedH/cp,coldArithmeticTolerance*fabs(expectedH/cp),"gas-only analytic bulk temperature");equal(values[6],0,"gas-only wall heat exactly zero");
}
printf("COLD_ENERGY maximum_abs_residual=%.17g maximum_rounding_bound=%.17g maximum_node_error=%.17g maximum_wall_energy_error=%.17g\n",largestColdResidual,largestColdBound,largestColdNodeError,largestColdWallError);
puts("PASS scalar/analytic cold-wall enthalpy, wall/gas energy, ring mass and rollback for the actual profile storage owner");
}
puts("PASS all frozen-implementation differential checks");return 0;}
