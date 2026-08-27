#ifndef UGKWP_GPU_PACKING_PROJECTION_ALGEBRA_CUH
#define UGKWP_GPU_PACKING_PROJECTION_ALGEBRA_CUH

__device__ void mobilePackingPrimitive
(
    const DeviceState& s,
    const int c,
    double& eps,
    double& ux,
    double& uy,
    double& uz
)
{
    if (c < 0 || c >= s.nCells)
    {
        eps = 0.0;
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        return;
    }
    const double rho = clampMin(finiteOr(s.mobilePackingRho[c], 0.0), 0.0);
    eps = rho/clampMin(s.rhoSolid, OfVSmall);
    if (rho <= s.epsSMin*s.rhoSolid)
    {
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        return;
    }
    ux = finiteOr(s.mobilePackingMomX[c], 0.0)/rho;
    uy = finiteOr(s.mobilePackingMomY[c], 0.0)/rho;
    uz = finiteOr(s.mobilePackingMomZ[c], 0.0)/rho;
}

__device__ double mobilePackingProjectedJacobiValue
(
    const DeviceState& s,
    const int c,
    const double* oldPressure
)
{
    double diagonal = 0.0;
    double neighbourSum = 0.0;
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
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        if (nei >= 0 && nei < s.nCells)
        {
            const int other = own == c ? nei : own;
            diagonal += a;
            neighbourSum +=
                a*clampMin(finiteOr(oldPressure[other], 0.0), 0.0);
        }
        else if (own == c && s.gasBoundaryKind[f] == 0)
        {
            diagonal += a;
        }
    }
    if (diagonal <= OfVSmall)
    {
        return 0.0;
    }

    const double rhs = clampRange
    (
        finiteOr(s.pressureDeltaEnergy[c], 0.0),
        -OfGreat,
        OfGreat
    );
    const double candidate =
        clampMin((rhs + neighbourSum)/diagonal, 0.0);
    const double relaxed =
        (1.0 - mobilePackingJacobiOmega)
       *clampMin(finiteOr(oldPressure[c], 0.0), 0.0)
      + mobilePackingJacobiOmega*candidate;
    return clampRange(finiteOr(relaxed, 0.0), 0.0, OfGreat);
}

__device__ void mobilePackingParticleVelocityCorrection
(
    const DeviceState& s,
    const int particleI,
    double& dux,
    double& duy,
    double& duz
)
{
    const int c = s.pCellId[particleI];
    dux = finiteOr(s.pressureDeltaMomX[c], 0.0);
    duy = finiteOr(s.pressureDeltaMomY[c], 0.0);
    duz = finiteOr(s.pressureDeltaMomZ[c], 0.0);

    int closestPlane = -1;
    double closestCoordinate = 1.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int plane = start + j;
        const double nx = s.planeNx[plane];
        const double ny = s.planeNy[plane];
        const double nz = s.planeNz[plane];
        const double centreToFaceDistance =
            s.planeD[plane]
          - (nx*s.Cx[c] + ny*s.Cy[c] + nz*s.Cz[c]);
        if (!(centreToFaceDistance > OfVSmall))
        {
            continue;
        }
        const double particleToFaceDistance =
            s.planeD[plane]
          - (nx*s.px[particleI] + ny*s.py[particleI] + nz*s.pz[particleI]);
        const double coordinate = clampRange
        (
            finiteOr(particleToFaceDistance/centreToFaceDistance, 1.0),
            0.0,
            1.0
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
    const double magSf = clampMin(finiteOr(s.magSf[faceI], 0.0), 0.0);
    if (magSf <= OfVSmall)
    {
        return;
    }
    const double nx = s.planeNx[closestPlane];
    const double ny = s.planeNy[closestPlane];
    const double nz = s.planeNz[closestPlane];
    const double sign = s.faceOwner[faceI] == c ? 1.0 : -1.0;
    const double faceVelocityCorrection =
        sign*finiteOr(s.solidPressurePhiEnergy[faceI], 0.0)/magSf;
    const double cellNormalCorrection = dux*nx + duy*ny + duz*nz;
    const double faceBlend = 1.0 - closestCoordinate;
    const double normalAdjustment =
        faceBlend*(faceVelocityCorrection - cellNormalCorrection);
    dux += normalAdjustment*nx;
    duy += normalAdjustment*ny;
    duz += normalAdjustment*nz;
}

#endif
