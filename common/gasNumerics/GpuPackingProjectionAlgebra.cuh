#include "GpuPrecisionTypes.H"
#ifndef UGKWP_GPU_PACKING_PROJECTION_ALGEBRA_CUH
#define UGKWP_GPU_PACKING_PROJECTION_ALGEBRA_CUH

__device__ void mobilePackingPrimitive
(
    const DeviceState& s,
    const int c,
    GpuReal& eps,
    GpuReal& ux,
    GpuReal& uy,
    GpuReal& uz
)
{
    if (c < 0 || c >= s.nCells)
    {
        eps = GPU_R(0.0);
        ux = GPU_R(0.0);
        uy = GPU_R(0.0);
        uz = GPU_R(0.0);
        return;
    }
    const GpuReal rho = clampMin(finiteOr(s.mobilePackingRho[c], GPU_R(0.0)), GPU_R(0.0));
    eps = rho/clampMin(s.rhoSolid, OfVSmall);
    if (rho <= s.epsSMin*s.rhoSolid)
    {
        ux = GPU_R(0.0);
        uy = GPU_R(0.0);
        uz = GPU_R(0.0);
        return;
    }
    ux = finiteOr(s.mobilePackingMomX[c], GPU_R(0.0))/rho;
    uy = finiteOr(s.mobilePackingMomY[c], GPU_R(0.0))/rho;
    uz = finiteOr(s.mobilePackingMomZ[c], GPU_R(0.0))/rho;
}

__device__ GpuReal mobilePackingProjectedJacobiValue
(
    const DeviceState& s,
    const int c,
    const GpuReal* oldPressure
)
{
    GpuReal diagonal = GPU_R(0.0);
    GpuReal neighbourSum = GPU_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const GpuReal a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], GPU_R(0.0)),
            GPU_R(0.0)
        );
        if (nei >= 0 && nei < s.nCells)
        {
            const int other = own == c ? nei : own;
            diagonal += a;
            neighbourSum +=
                a*clampMin(finiteOr(oldPressure[other], GPU_R(0.0)), GPU_R(0.0));
        }
        else if (own == c && s.gasBoundaryKind[f] == 0)
        {
            diagonal += a;
        }
    }
    if (diagonal <= OfVSmall)
    {
        return GPU_R(0.0);
    }

    const GpuReal rhs = clampRange
    (
        finiteOr(s.pressureDeltaEnergy[c], GPU_R(0.0)),
        -OfGreat,
        OfGreat
    );
    const GpuReal candidate =
        clampMin((rhs + neighbourSum)/diagonal, GPU_R(0.0));
    const GpuReal relaxed =
        (GPU_R(1.0) - mobilePackingJacobiOmega)
       *clampMin(finiteOr(oldPressure[c], GPU_R(0.0)), GPU_R(0.0))
      + mobilePackingJacobiOmega*candidate;
    return clampRange(finiteOr(relaxed, GPU_R(0.0)), GPU_R(0.0), OfGreat);
}

__device__ void mobilePackingParticleVelocityCorrection
(
    const DeviceState& s,
    const int particleI,
    GpuReal& dux,
    GpuReal& duy,
    GpuReal& duz
)
{
    const int c = s.pCellId[particleI];
    dux = finiteOr(s.pressureDeltaMomX[c], GPU_R(0.0));
    duy = finiteOr(s.pressureDeltaMomY[c], GPU_R(0.0));
    duz = finiteOr(s.pressureDeltaMomZ[c], GPU_R(0.0));

    int closestPlane = -1;
    GpuReal closestCoordinate = GPU_R(1.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int plane = start + j;
        const GpuReal nx = s.planeNx[plane];
        const GpuReal ny = s.planeNy[plane];
        const GpuReal nz = s.planeNz[plane];
        const GpuReal centreToFaceDistance =
            s.planeD[plane]
          - (nx*s.Cx[c] + ny*s.Cy[c] + nz*s.Cz[c]);
        if (!(centreToFaceDistance > OfVSmall))
        {
            continue;
        }
        const GpuReal particleToFaceDistance =
            s.planeD[plane]
          - (nx*s.px[particleI] + ny*s.py[particleI] + nz*s.pz[particleI]);
        const GpuReal coordinate = clampRange
        (
            finiteOr(particleToFaceDistance/centreToFaceDistance, GPU_R(1.0)),
            GPU_R(0.0),
            GPU_R(1.0)
        );
        if (coordinate < closestCoordinate)
        {
            closestCoordinate = coordinate;
            closestPlane = plane;
        }
    }
    if (closestPlane < 0)
    {
        return;
    }

    const int faceI = s.cellFaceId[closestPlane];
    if (faceI < 0 || faceI >= s.nFaces)
    {
        return;
    }
    const GpuReal magSf = clampMin(finiteOr(s.magSf[faceI], GPU_R(0.0)), GPU_R(0.0));
    if (magSf <= OfVSmall)
    {
        return;
    }
    const GpuReal nx = s.planeNx[closestPlane];
    const GpuReal ny = s.planeNy[closestPlane];
    const GpuReal nz = s.planeNz[closestPlane];
    const GpuReal sign = s.faceOwner[faceI] == c ? GPU_R(1.0) : -GPU_R(1.0);
    const GpuReal faceVelocityCorrection =
        sign*finiteOr(s.solidPressurePhiEnergy[faceI], GPU_R(0.0))/magSf;
    const GpuReal cellNormalCorrection = dux*nx + duy*ny + duz*nz;
    const GpuReal faceBlend = GPU_R(1.0) - closestCoordinate;
    const GpuReal normalAdjustment =
        faceBlend*(faceVelocityCorrection - cellNormalCorrection);
    dux += normalAdjustment*nx;
    duy += normalAdjustment*ny;
    duz += normalAdjustment*nz;
}

#endif
