#ifndef UGKWP_GPU_PACKING_PROJECTION_COOPERATIVE_CUH
#define UGKWP_GPU_PACKING_PROJECTION_COOPERATIVE_CUH

#include <cooperative_groups.h>

namespace packingCg = cooperative_groups;

__device__ void expandMobilePackingActivityFrontierDevice
(
    DeviceState& s,
    const int frontierI,
    const int* frontier,
    int* nextFrontier,
    int* nextFrontierCount,
    int* accumulatedCells,
    int* accumulatedCount,
    int* activityMask
)
{
    const int c = frontier[frontierI];
    if (c < 0 || c >= s.nCells)
    {
        return;
    }
    const int start = s.cellPlaneStart[c];
    const int nCellFaces = s.cellPlaneCount[c];
    for (int j = 0; j < nCellFaces; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            continue;
        }
        const int other = own == c ? nei : own;
        if (other < 0 || other >= s.nCells)
        {
            continue;
        }
        if (atomicCAS(&activityMask[other], 0, 1) != 0)
        {
            continue;
        }
        const int nextSlot = atomicAdd(nextFrontierCount, 1);
        const int accumulatedSlot = atomicAdd(accumulatedCount, 1);
        if (nextSlot < s.nCells)
        {
            nextFrontier[nextSlot] = other;
        }
        if (accumulatedSlot < s.nCells)
        {
            accumulatedCells[accumulatedSlot] = other;
        }
    }
}

__device__ void initialiseMobilePackingCorrectionRegionDevice
(
    DeviceState& s,
    const int activeI,
    const int count
)
{
    const int c = s.mobilePackingActiveCellList[activeI];
    if (c < 0 || c >= s.nCells)
    {
        return;
    }
    if (activeI == 0)
    {
        *s.mobilePackingCorrectionCellCount = count;
    }
    s.mobilePackingCorrectionCellMask[c] = 1;
    s.mobilePackingCorrectionCellList[activeI] = c;
}

__device__ void solveActiveMobilePackingPressureJacobiDevice
(
    DeviceState& s,
    const int activeI,
    const int* activeCells,
    const double* oldPressure,
    double* newPressure
)
{
    const int c = activeCells[activeI];
    newPressure[c] = mobilePackingProjectedJacobiValue(s, c, oldPressure);
}

__device__ void computeMobilePackingFaceCorrectionFluxDevice
(
    DeviceState& s,
    const int f,
    const double* pressure,
    const double dt
)
{
    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    const bool ownerPressureActive =
        own >= 0 && own < s.nCells
     && s.mobilePackingActiveCellMask[own] != 0;
    const bool neighbourPressureActive =
        nei >= 0 && nei < s.nCells
     && s.mobilePackingActiveCellMask[nei] != 0;
    if (!ownerPressureActive && !neighbourPressureActive)
    {
        s.solidPressurePhiMomX[f] = 0.0;
        s.solidPressurePhiMomY[f] = 0.0;
        s.solidPressurePhiMomZ[f] = 0.0;
        s.solidPressurePhiEnergy[f] = 0.0;
        return;
    }
    const double invRhoSolid = 1.0/clampMin(s.rhoSolid, OfVSmall);
    double solidVolumeFlux = 0.0;
    double faceMobileFraction = 0.0;
    double velocityCorrectionFlux = 0.0;

    if (own >= 0 && own < s.nCells && nei >= 0 && nei < s.nCells)
    {
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        const double pressureOwn =
            clampMin(finiteOr(pressure[own], 0.0), 0.0);
        const double pressureNei =
            clampMin(finiteOr(pressure[nei], 0.0), 0.0);
        const double epsOwn = clampMin
        (
            finiteOr(s.mobilePackingRho[own], 0.0)*invRhoSolid,
            0.0
        );
        const double epsNei = clampMin
        (
            finiteOr(s.mobilePackingRho[nei], 0.0)*invRhoSolid,
            0.0
        );
        const double w =
            clampRange(finiteOr(s.faceWeight[f], 0.5), 0.0, 1.0);
        faceMobileFraction = clampMin
        (
            w*epsOwn + (1.0 - w)*epsNei,
            s.epsSMin
        );
        solidVolumeFlux = finiteOr
        (
            dt*invRhoSolid*a*(pressureOwn - pressureNei),
            0.0
        );
        velocityCorrectionFlux = finiteOr
        (
            solidVolumeFlux/faceMobileFraction,
            0.0
        );
    }
    else if
    (
        own >= 0
     && own < s.nCells
     && s.gasBoundaryKind[f] == 0
    )
    {
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        const double pressureOwn =
            clampMin(finiteOr(pressure[own], 0.0), 0.0);
        faceMobileFraction = clampMin
        (
            finiteOr(s.mobilePackingRho[own], 0.0)*invRhoSolid,
            s.epsSMin
        );
        solidVolumeFlux = finiteOr
        (
            dt*invRhoSolid*a*pressureOwn,
            0.0
        );
        velocityCorrectionFlux = finiteOr
        (
            solidVolumeFlux/faceMobileFraction,
            0.0
        );
    }

    s.solidPressurePhiMomX[f] = solidVolumeFlux;
    s.solidPressurePhiMomY[f] = faceMobileFraction;
    s.solidPressurePhiMomZ[f] = 0.0;
    s.solidPressurePhiEnergy[f] = velocityCorrectionFlux;
}

__device__ void reconstructMobilePackingVelocityCorrectionDevice
(
    DeviceState& s,
    const int activeI,
    const int* correctionCells
)
{
    const int c = correctionCells[activeI];
    double m00 = 0.0;
    double m01 = 0.0;
    double m02 = 0.0;
    double m11 = 0.0;
    double m12 = 0.0;
    double m22 = 0.0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double magSf = clampMin(finiteOr(s.magSf[f], 0.0), 0.0);
        if (magSf <= OfVSmall)
        {
            continue;
        }
        const double nx = s.Sfx[f]/magSf;
        const double ny = s.Sfy[f]/magSf;
        const double nz = s.Sfz[f]/magSf;
        const double velocityCorrectionFlux =
            finiteOr(s.solidPressurePhiEnergy[f], 0.0);
        m00 += nx*s.Sfx[f];
        m01 += nx*s.Sfy[f];
        m02 += nx*s.Sfz[f];
        m11 += ny*s.Sfy[f];
        m12 += ny*s.Sfz[f];
        m22 += nz*s.Sfz[f];
        b0 += nx*velocityCorrectionFlux;
        b1 += ny*velocityCorrectionFlux;
        b2 += nz*velocityCorrectionFlux;
    }

    const double trace = m00 + m11 + m22;
    if (!(trace > OfVSmall) || !finiteDevice(trace))
    {
        s.pressureDeltaMomX[c] = 0.0;
        s.pressureDeltaMomY[c] = 0.0;
        s.pressureDeltaMomZ[c] = 0.0;
        return;
    }
    const double regularisation = 1.0e-12*trace;
    const double l00 = sqrt(fmax(m00 + regularisation, regularisation));
    const double l10 = m01/l00;
    const double l20 = m02/l00;
    const double l11 = sqrt
    (
        fmax(m11 + regularisation - l10*l10, regularisation)
    );
    const double l21 = (m12 - l20*l10)/l11;
    const double l22 = sqrt
    (
        fmax
        (
            m22 + regularisation - l20*l20 - l21*l21,
            regularisation
        )
    );

    const double y0 = b0/l00;
    const double y1 = (b1 - l10*y0)/l11;
    const double y2 = (b2 - l20*y0 - l21*y1)/l22;
    const double uz = y2/l22;
    const double uy = (y1 - l21*uz)/l11;
    const double ux = (y0 - l10*uy - l20*uz)/l00;
    s.pressureDeltaMomX[c] = finiteOr(ux, 0.0);
    s.pressureDeltaMomY[c] = finiteOr(uy, 0.0);
    s.pressureDeltaMomZ[c] = finiteOr(uz, 0.0);
}

__device__ void applyMobilePackingCorrectionToParticleDevice
(
    DeviceState& s,
    const int i
)
{
    if (!mobilePackingParticleEligible(s, i))
    {
        return;
    }
    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        return;
    }
    if (s.mobilePackingCorrectionCellMask[c] == 0)
    {
        return;
    }
    double dux = 0.0;
    double duy = 0.0;
    double duz = 0.0;
    mobilePackingParticleVelocityCorrection(s, i, dux, duy, duz);
    s.pux[i] = finiteOr(s.pux[i], 0.0) + dux;
    s.puy[i] = finiteOr(s.puy[i], 0.0) + duy;
    s.puz[i] = finiteOr(s.puz[i], 0.0) + duz;
    s.puxOld[i] = finiteOr(s.puxOld[i], 0.0) + dux;
    s.puyOld[i] = finiteOr(s.puyOld[i], 0.0) + duy;
    s.puzOld[i] = finiteOr(s.puzOld[i], 0.0) + duz;
}

__global__ void completeMobilePackingProjectionCooperativeKernel
(
    DeviceState* sp,
    const double dt
)
{
    packingCg::grid_group grid = packingCg::this_grid();
    DeviceState& s = *sp;
    const int globalThread = blockIdx.x*blockDim.x + threadIdx.x;
    const int globalStride = blockDim.x*gridDim.x;
    const int seedCount =
        checkedMobilePackingCount(s, s.mobilePackingActiveCellCount);
    if (seedCount == 0)
    {
        return;
    }

    int* currentFrontier = s.mobilePackingFrontierCurrent;
    int* nextFrontier = s.mobilePackingFrontierNext;
    int* currentFrontierCount = s.mobilePackingFrontierCurrentCount;
    int* nextFrontierCount = s.mobilePackingFrontierNextCount;
    for (int layer = 0; layer < s.packingProjectionIterations; ++layer)
    {
        if (globalThread == 0)
        {
            *nextFrontierCount = 0;
        }
        grid.sync();
        const int frontierCount =
            checkedMobilePackingCount(s, currentFrontierCount);
        for
        (
            int frontierI = globalThread;
            frontierI < frontierCount;
            frontierI += globalStride
        )
        {
            expandMobilePackingActivityFrontierDevice
            (
                s,
                frontierI,
                currentFrontier,
                nextFrontier,
                nextFrontierCount,
                s.mobilePackingActiveCellList,
                s.mobilePackingActiveCellCount,
                s.mobilePackingActiveCellMask
            );
        }
        grid.sync();
        int* frontierSwap = currentFrontier;
        currentFrontier = nextFrontier;
        nextFrontier = frontierSwap;
        int* countSwap = currentFrontierCount;
        currentFrontierCount = nextFrontierCount;
        nextFrontierCount = countSwap;
    }

    double* oldPressure = s.collisionalPressure;
    double* newPressure = s.pressureKickScale;
    for (int iter = 0; iter < s.packingProjectionIterations; ++iter)
    {
        const int activeCount =
            checkedMobilePackingCount(s, s.mobilePackingActiveCellCount);
        for
        (
            int activeI = globalThread;
            activeI < activeCount;
            activeI += globalStride
        )
        {
            solveActiveMobilePackingPressureJacobiDevice
            (
                s,
                activeI,
                s.mobilePackingActiveCellList,
                oldPressure,
                newPressure
            );
        }
        grid.sync();
        double* pressureSwap = oldPressure;
        oldPressure = newPressure;
        newPressure = pressureSwap;
    }

    const int pressureActiveCount =
        checkedMobilePackingCount(s, s.mobilePackingActiveCellCount);
    for
    (
        int activeI = globalThread;
        activeI < pressureActiveCount;
        activeI += globalStride
    )
    {
        initialiseMobilePackingCorrectionRegionDevice
        (
            s,
            activeI,
            pressureActiveCount
        );
    }
    grid.sync();
    if (globalThread == 0)
    {
        *s.mobilePackingFrontierNextCount = 0;
    }
    grid.sync();
    for
    (
        int activeI = globalThread;
        activeI < pressureActiveCount;
        activeI += globalStride
    )
    {
        expandMobilePackingActivityFrontierDevice
        (
            s,
            activeI,
            s.mobilePackingActiveCellList,
            s.mobilePackingFrontierNext,
            s.mobilePackingFrontierNextCount,
            s.mobilePackingCorrectionCellList,
            s.mobilePackingCorrectionCellCount,
            s.mobilePackingCorrectionCellMask
        );
    }
    grid.sync();

    for
    (
        int f = globalThread;
        f < s.nFaces;
        f += globalStride
    )
    {
        computeMobilePackingFaceCorrectionFluxDevice(s, f, oldPressure, dt);
    }
    grid.sync();

    const int correctionCount =
        checkedMobilePackingCount(s, s.mobilePackingCorrectionCellCount);
    if (correctionCount < pressureActiveCount)
    {
        asm("trap;");
    }
    for
    (
        int activeI = globalThread;
        activeI < correctionCount;
        activeI += globalStride
    )
    {
        reconstructMobilePackingVelocityCorrectionDevice
        (
            s,
            activeI,
            s.mobilePackingCorrectionCellList
        );
    }
    grid.sync();

    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = globalThread;
        i < nParticles;
        i += globalStride
    )
    {
        applyMobilePackingCorrectionToParticleDevice(s, i);
    }
}

#endif
