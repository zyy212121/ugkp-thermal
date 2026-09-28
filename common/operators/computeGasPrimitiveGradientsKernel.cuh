#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeGasPrimitiveGradientsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL grx = GPU_OPERATOR_R(0.0), gry = GPU_OPERATOR_R(0.0), grz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guxx = GPU_OPERATOR_R(0.0), guxy = GPU_OPERATOR_R(0.0), guxz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guyx = GPU_OPERATOR_R(0.0), guyy = GPU_OPERATOR_R(0.0), guyz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guzx = GPU_OPERATOR_R(0.0), guzy = GPU_OPERATOR_R(0.0), guzz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gpx = GPU_OPERATOR_R(0.0), gpy = GPU_OPERATOR_R(0.0), gpz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gtx = GPU_OPERATOR_R(0.0), gty = GPU_OPERATOR_R(0.0), gtz = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        const GPU_OPERATOR_REAL sx = sign*s.Sfx[f];
        const GPU_OPERATOR_REAL sy = sign*s.Sfy[f];
        const GPU_OPERATOR_REAL sz = sign*s.Sfz[f];
        const GasPrimDevice qf =
            riemannFacePrimitiveForGradient(s, c, f);
        grx += qf.rho*sx; gry += qf.rho*sy; grz += qf.rho*sz;
        guxx += qf.ux*sx; guxy += qf.ux*sy; guxz += qf.ux*sz;
        guyx += qf.uy*sx; guyy += qf.uy*sy; guyz += qf.uy*sz;
        guzx += qf.uz*sx; guzy += qf.uz*sy; guzz += qf.uz*sz;
        gpx += qf.p*sx; gpy += qf.p*sy; gpz += qf.p*sz;
        gtx += qf.T*sx; gty += qf.T*sy; gtz += qf.T*sz;
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    s.gradRhoX[c] = grx*invV; s.gradRhoY[c] = gry*invV; s.gradRhoZ[c] = grz*invV;
    s.gradUxX[c] = guxx*invV; s.gradUxY[c] = guxy*invV; s.gradUxZ[c] = guxz*invV;
    s.gradUyX[c] = guyx*invV; s.gradUyY[c] = guyy*invV; s.gradUyZ[c] = guyz*invV;
    s.gradUzX[c] = guzx*invV; s.gradUzY[c] = guzy*invV; s.gradUzZ[c] = guzz*invV;
    s.gradPx[c] = gpx*invV; s.gradPy[c] = gpy*invV; s.gradPz[c] = gpz*invV;
    s.gradTX[c] = gtx*invV; s.gradTY[c] = gty*invV; s.gradTZ[c] = gtz*invV;
}

__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue
(
    const DeviceState& s,
    const int f,
    const int owner
)
{
    const GPU_OPERATOR_REAL rhoSafe = clampMin(riemannFacePrimitiveForGradient(s, owner, f).rho, s.rhoMin);
    const GPU_OPERATOR_REAL nu = s.gasMu/rhoSafe;
    const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL dux = s.Ux[owner] - wallUx;
    const GPU_OPERATOR_REAL duy = s.Uy[owner] - wallUy;
    const GPU_OPERATOR_REAL duz = s.Uz[owner] - wallUz;
    const GPU_OPERATOR_REAL y = clampMin(s.sstWallDistance[owner], OfVSmall);
    const GPU_OPERATOR_REAL magGradU = sqrt(dux*dux + duy*duy + duz*duz)/y;
    return ugkpwall::omegaWallFunctionState
    (
        s.k[owner],
        magGradU,
        y,
        nu,
        s.sstCoefficients.beta1,
        s.sstWallCmu,
        s.sstWallKappa,
        s.sstWallE
    ).omega;
}

__device__ GPU_OPERATOR_REAL sstBoundaryValue
(
    const DeviceState& s,
    const int f,
    const int owner,
    const bool omegaField
)
{
    const GPU_OPERATOR_REAL centre = omegaField ? s.omega[owner] : s.k[owner];
    const int boundaryKind = s.riemannBoundaryKind[f];
    if (boundaryKind == 2)
    {
        if (!omegaField)
        {
            return s.sstWallTreatment == 0 ? GPU_OPERATOR_R(0.0) : centre;
        }
        if (s.sstWallTreatment == 0)
        {
            const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[owner], s.rhoMin);
            return ugkwp::sstLowReWallOmega
            (
                s.gasMu/rhoSafe,
                s.sstWallDistance[owner],
                s.sstCoefficients
            );
        }
        return centre;
    }
    if (boundaryKind == 1 || boundaryKind == 3 || boundaryKind == 4)
    {
        return centre;
    }

    const int mode = omegaField
      ? s.sstBoundaryOmegaMode[f] : s.sstBoundaryKMode[f];
    const GPU_OPERATOR_REAL prescribed = omegaField
      ? s.sstBoundaryOmega[f] : s.sstBoundaryK[f];
    if (mode == 1)
    {
        return prescribed;
    }
    if (mode == 2)
    {
        const GPU_OPERATOR_REAL outwardMassDirection =
            s.Ux[owner]*s.Sfx[f]
          + s.Uy[owner]*s.Sfy[f]
          + s.Uz[owner]*s.Sfz[f];
        return outwardMassDirection >= GPU_OPERATOR_R(0.0) ? centre : prescribed;
    }
    return centre;
}

__global__ void applySstWallFunctionStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if
    (
        c >= s.nCells
     || s.sstConfigured == 0
     || s.sstWallTreatment != 1
    )
    {
        return;
    }

    GPU_OPERATOR_REAL omegaSum = GPU_OPERATOR_R(0.0);
    int wallCount = 0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f >= s.nInternalFaces
         && f < s.nFaces
         && s.riemannBoundaryKind[f] == 2
        )
        {
            omegaSum += sstDynamicOmegaWallValue(s, f, c);
            ++wallCount;
        }
    }
    if (wallCount > 0)
    {
        const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
        const GPU_OPERATOR_REAL omegaTarget = clampMin
        (
            finiteOr(omegaSum/GPU_OPERATOR_REAL(wallCount), s.sstOmegaMin),
            s.sstOmegaMin
        );
        s.omega[c] = omegaTarget;
        s.rhoOmega[c] = rhoSafe*omegaTarget;
    }
}

__global__ void initialiseSstConservativeStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c == 0 && s.sstConfigured != 0)
    {
        const GPU_OPERATOR_REAL PrRatio =
            s.gasPrClamped/clampMin(s.turbulentPrandtl, OfSmall);
        s.sstJayatillekeP = ugkpwall::jayatillekeSmoothP(PrRatio);
        s.sstThermalYPlus = ugkpwall::jayatillekeThermalYPlus
        (
            PrRatio,
            s.sstWallKappa,
            s.sstWallE
        );
    }
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL kSafe = clampMin(finiteOr(s.k[c], s.sstKMin), s.sstKMin);
    const GPU_OPERATOR_REAL omegaSafe =
        clampMin(finiteOr(s.omega[c], s.sstOmegaMin), s.sstOmegaMin);
    s.k[c] = kSafe;
    s.omega[c] = omegaSafe;
    s.rhoK[c] = rhoSafe*kSafe;
    s.rhoOmega[c] = rhoSafe*omegaSafe;
    s.rhoKInitial[c] = s.rhoK[c];
    s.rhoOmegaInitial[c] = s.rhoOmega[c];
    s.nut[c] = GPU_OPERATOR_R(0.0);
}

__global__ void recoverSstPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.rhoK[c] = clampMin(finiteOr(s.rhoK[c], rhoKFloor), rhoKFloor);
    s.rhoOmega[c] =
        clampMin(finiteOr(s.rhoOmega[c], rhoOmegaFloor), rhoOmegaFloor);
    s.k[c] = s.rhoK[c]/rhoSafe;
    s.omega[c] = s.rhoOmega[c]/rhoSafe;
}
