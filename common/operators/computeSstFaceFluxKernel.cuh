#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeSstFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces || s.sstConfigured == 0)
    {
        return;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        s.sstPhiRhoK[f] = GPU_OPERATOR_R(0.0);
        s.sstPhiRhoOmega[f] = GPU_OPERATOR_R(0.0);
        return;
    }
    const int nei = coupledFaceNeighbour(s, f);
    const int boundaryKind = nei >= 0 ? 0 : s.riemannBoundaryKind[f];
    GPU_OPERATOR_REAL sstMappedNeiCx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL sstMappedNeiCy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL sstMappedNeiCz = GPU_OPERATOR_R(0.0);
    if (nei >= 0)
    {
        periodicMappedCellCentre
        (
            s, f, nei,
            sstMappedNeiCx, sstMappedNeiCy, sstMappedNeiCz
        );
    }
    if (boundaryKind == 1 || boundaryKind == 3 || boundaryKind == 4)
    {
        s.sstPhiRhoK[f] = GPU_OPERATOR_R(0.0);
        s.sstPhiRhoOmega[f] = GPU_OPERATOR_R(0.0);
        return;
    }

    const GPU_OPERATOR_REAL massFlux = s.gasPhiRho[f];
    const GPU_OPERATOR_REAL ownerWeight = clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    const GPU_OPERATOR_REAL kExterior = nei >= 0
      ? s.k[nei] : sstBoundaryValue(s, f, own, false);
    const GPU_OPERATOR_REAL omegaExterior = nei >= 0
      ? s.omega[nei] : sstBoundaryValue(s, f, own, true);
    const GPU_OPERATOR_REAL kUpwind = massFlux >= GPU_OPERATOR_R(0.0) ? s.k[own] : kExterior;
    const GPU_OPERATOR_REAL omegaUpwind =
        massFlux >= GPU_OPERATOR_R(0.0) ? s.omega[own] : omegaExterior;

    const GPU_OPERATOR_REAL rhoFace = nei >= 0
      ? ownerWeight*s.rho[own] + (GPU_OPERATOR_R(1.0) - ownerWeight)*s.rho[nei]
      : (boundaryKind == 2 ? riemannFacePrimitiveForGradient(s, own, f).rho : s.rho[own]);
    const GPU_OPERATOR_REAL f1Face = nei >= 0
      ? ownerWeight*s.sstF1[own] + (GPU_OPERATOR_R(1.0) - ownerWeight)*s.sstF1[nei]
      : s.sstF1[own];
    const GPU_OPERATOR_REAL nuFace = s.gasMu/clampMin(rhoFace, s.rhoMin);
    GPU_OPERATOR_REAL nutFace = nei >= 0
      ? ownerWeight*s.nut[own] + (GPU_OPERATOR_R(1.0) - ownerWeight)*s.nut[nei]
      : s.nut[own];
    if (boundaryKind == 2)
    {
        nutFace = GPU_OPERATOR_R(0.0);
        if (s.sstWallTreatment == 1)
        {
            const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL dux = s.Ux[own] - wallUx;
            const GPU_OPERATOR_REAL duy = s.Uy[own] - wallUy;
            const GPU_OPERATOR_REAL duz = s.Uz[own] - wallUz;
            nutFace = ugkpwall::spaldingWallState
            (
                sqrt(dux*dux + duy*duy + duz*duz),
                s.sstWallDistance[own],
                nuFace,
                s.sstWallKappa,
                s.sstWallE
            ).nut;
        }
    }
    const GPU_OPERATOR_REAL dk =
        nuFace + ugkwp::sstAlphaK(f1Face, s.sstCoefficients)*nutFace;
    const GPU_OPERATOR_REAL domega =
        nuFace
      + ugkwp::sstAlphaOmega(f1Face, s.sstCoefficients)*nutFace;

    GPU_OPERATOR_REAL snGradK = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL snGradOmega = GPU_OPERATOR_R(0.0);
    if (nei >= 0)
    {
        const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
        const ugkptransport::SnGradGeometry geometry =
            ugkptransport::makeInternalSnGradGeometry
            (
                ugkptransport::Vector3{s.Cx[own], s.Cy[own], s.Cz[own]},
                ugkptransport::Vector3
                {
                    sstMappedNeiCx, sstMappedNeiCy, sstMappedNeiCz
                },
                ugkptransport::Vector3
                {
                    s.Sfx[f]/area,
                    s.Sfy[f]/area,
                    s.Sfz[f]/area
                }
            );
        snGradK = ugkptransport::correctedSnGrad
        (
            s.k[own],
            s.k[nei],
            ugkptransport::Vector3
            {
                s.gradKX[own], s.gradKY[own], s.gradKZ[own]
            },
            ugkptransport::Vector3
            {
                s.gradKX[nei], s.gradKY[nei], s.gradKZ[nei]
            },
            ownerWeight,
            geometry
        );
        snGradOmega = ugkptransport::correctedSnGrad
        (
            s.omega[own],
            s.omega[nei],
            ugkptransport::Vector3
            {
                s.gradOmegaX[own], s.gradOmegaY[own], s.gradOmegaZ[own]
            },
            ugkptransport::Vector3
            {
                s.gradOmegaX[nei], s.gradOmegaY[nei], s.gradOmegaZ[nei]
            },
            ownerWeight,
            geometry
        );
    }
    else
    {
        const int kMode = s.sstBoundaryKMode[f];
        const int omegaMode = s.sstBoundaryOmegaMode[f];
        const bool kFixed =
            (boundaryKind == 2 && s.sstWallTreatment == 0)
          || kMode == 1
          || (kMode == 2 && massFlux < GPU_OPERATOR_R(0.0));
        const bool omegaFixed = boundaryKind == 2 || omegaMode == 1
          || (omegaMode == 2 && massFlux < GPU_OPERATOR_R(0.0));
        if (kFixed)
        {
            snGradK = s.deltaCoeffs[f]*(kExterior - s.k[own]);
        }
        if (omegaFixed)
        {
            snGradOmega =
                s.deltaCoeffs[f]*(omegaExterior - s.omega[own]);
        }
    }

    const GPU_OPERATOR_REAL area = s.magSf[f];
    s.sstPhiRhoK[f] =
        massFlux*kUpwind - rhoFace*dk*snGradK*area;
    s.sstPhiRhoOmega[f] =
        massFlux*omegaUpwind - rhoFace*domega*snGradOmega*area;
}

__global__ void enforcePeriodicSstFluxAntisymmetryKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (!isPeriodicFace(s, f))
    {
        return;
    }
    const int pair = s.facePeriodicPair[f];
    if (f > pair)
    {
        return;
    }
    const GPU_OPERATOR_REAL rhoK = GPU_OPERATOR_R(0.5)*(s.sstPhiRhoK[f] - s.sstPhiRhoK[pair]);
    const GPU_OPERATOR_REAL rhoOmega =
        GPU_OPERATOR_R(0.5)*(s.sstPhiRhoOmega[f] - s.sstPhiRhoOmega[pair]);
    s.sstPhiRhoK[f] = rhoK;
    s.sstPhiRhoOmega[f] = rhoOmega;
    s.sstPhiRhoK[pair] = -rhoK;
    s.sstPhiRhoOmega[pair] = -rhoOmega;
}

__device__ GPU_OPERATOR_REAL sstKProductionForCell
(
    const DeviceState& s,
    const int c,
    const GPU_OPERATOR_REAL gByNu
)
{
    GPU_OPERATOR_REAL production = ugkwp::sstKProduction
    (
        s.k[c],
        s.omega[c],
        s.nut[c],
        gByNu,
        s.sstCoefficients
    );
    if (s.sstWallTreatment != 1)
    {
        return production;
    }

    GPU_OPERATOR_REAL wallProductionSum = GPU_OPERATOR_R(0.0);
    int wallCount = 0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < s.nInternalFaces
         || f >= s.nFaces
         || s.riemannBoundaryKind[f] != 2
        )
        {
            continue;
        }
        const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL dux = s.Ux[c] - wallUx;
        const GPU_OPERATOR_REAL duy = s.Uy[c] - wallUy;
        const GPU_OPERATOR_REAL duz = s.Uz[c] - wallUz;
        const GPU_OPERATOR_REAL y = clampMin(s.sstWallDistance[c], OfVSmall);
        const GPU_OPERATOR_REAL magGradU = sqrt(dux*dux + duy*duy + duz*duz)/y;
        wallProductionSum += ugkpwall::omegaWallFunctionState
        (
            s.k[c],
            magGradU,
            y,
            s.gasMu/clampMin(riemannFacePrimitiveForGradient(s, c, f).rho, s.rhoMin),
            s.sstCoefficients.beta1,
            s.sstWallCmu,
            s.sstWallKappa,
            s.sstWallE,
            production
        ).production;
        ++wallCount;
    }
    return wallCount > 0
      ? wallProductionSum/GPU_OPERATOR_REAL(wallCount)
      : production;
}

__global__ void applySstFluxAndSourceKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL fluxK = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL fluxOmega = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? -GPU_OPERATOR_R(1.0) : GPU_OPERATOR_R(1.0);
        fluxK += sign*s.sstPhiRhoK[f];
        fluxOmega += sign*s.sstPhiRhoOmega[f];
    }

    GPU_OPERATOR_REAL divU = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL s2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gByNu = GPU_OPERATOR_R(0.0);
    sstVelocityInvariants(s, c, divU, s2, gByNu);
    const GPU_OPERATOR_REAL gradDot =
        s.gradKX[c]*s.gradOmegaX[c]
      + s.gradKY[c]*s.gradOmegaY[c]
      + s.gradKZ[c]*s.gradOmegaZ[c];
    const GPU_OPERATOR_REAL cd = ugkwp::sstCrossDiffusion
    (
        s.omega[c],
        gradDot,
        s.sstCoefficients
    );
    const GPU_OPERATOR_REAL kProduction = sstKProductionForCell(s, c, gByNu);
    const GPU_OPERATOR_REAL sourceK = s.rho[c]*
    (
        kProduction
      - (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    const GPU_OPERATOR_REAL sourceOmega = ugkwp::sstOmegaSource
    (
        s.rho[c],
        s.k[c],
        s.omega[c],
        divU,
        gByNu,
        s2,
        s.sstF1[c],
        s.sstF2[c],
        cd,
        s.sstCoefficients
    );
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + sourceK);
    const GPU_OPERATOR_REAL deltaRhoOmega = dt*(fluxOmega*invV + sourceOmega);
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.sstSourceNumber[c] = fmax
    (
        fabs(dt*sourceK)/clampMin(s.rhoK[c], rhoKFloor),
        fabs(dt*sourceOmega)/clampMin(s.rhoOmega[c], rhoOmegaFloor)
    );
    s.rhoK[c] =
        clampMin(finiteOr(s.rhoK[c] + deltaRhoK, rhoKFloor), rhoKFloor);
    s.rhoOmega[c] = clampMin
    (
        finiteOr(s.rhoOmega[c] + deltaRhoOmega, rhoOmegaFloor),
        rhoOmegaFloor
    );
}
__global__ void computeGasCourantFieldKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    (void)dt;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        s.gasPhiRho[f] = OfGreat;
        return;
    }
    if
    (
        f >= s.nInternalFaces
     &&
        (
            s.riemannBoundaryKind[f] == 1
         || s.riemannBoundaryKind[f] == 3
         || s.riemannBoundaryKind[f] == 4
        )
    )
    {
        s.gasPhiRho[f] = GPU_OPERATOR_R(0.0);
        return;
    }

    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
    const GPU_OPERATOR_REAL nx = s.Sfx[f]/area;
    const GPU_OPERATOR_REAL ny = s.Sfy[f]/area;
    const GPU_OPERATOR_REAL nz = s.Sfz[f]/area;
    const GasPrimDevice left = makeGasPrimDevice
    (
        s.rho[own],
        s.Ux[own],
        s.Uy[own],
        s.Uz[own],
        s.p[own],
        s.Rgas,
        s.rhoMin,
        s.TgasMin
    );
    GasPrimDevice right = left;
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            s.gasPhiRho[f] = OfGreat;
            return;
        }
        right = makeGasPrimDevice
        (
            s.rho[nei],
            s.Ux[nei],
            s.Uy[nei],
            s.Uz[nei],
            s.p[nei],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }
    else if
    (
        s.riemannBoundaryKind[f] != 1
     && s.riemannBoundaryKind[f] != 2
    )
    {
        right = riemannBoundaryState(s, f, left);
    }
    const GPU_OPERATOR_REAL unLeft =
        left.ux*nx + left.uy*ny + left.uz*nz;
    const GPU_OPERATOR_REAL unRight =
        right.ux*nx + right.uy*ny + right.uz*nz;
    const GPU_OPERATOR_REAL aLeft =
        sqrt(clampMin(s.gammaGas*left.p/left.rho, OfSmall));
    const GPU_OPERATOR_REAL aRight =
        sqrt(clampMin(s.gammaGas*right.p/right.rho, OfSmall));
    const GPU_OPERATOR_REAL spectralRadius = fmax
    (
        fabs(unLeft) + aLeft,
        fabs(unRight) + aRight
    );
    const GPU_OPERATOR_REAL amaxSf = spectralRadius*area;

    s.gasPhiRho[f] = finiteDevice(amaxSf) ? amaxSf : OfGreat;
}

__global__ void computeGasConvectiveCourantByCellKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL sumAmaxSf = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        sumAmaxSf += finiteOr(s.gasPhiRho[f], OfGreat);
    }

                                                     
                                              
    const GPU_OPERATOR_REAL co =
        GPU_OPERATOR_R(0.5)*dt*sumAmaxSf/clampMin(s.V[c], OfSmall);
    s.gasFluxPositivityScale[c] = finiteDevice(co) ? co : OfGreat;
}

__global__ void computeGasDiffusionNumberKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    GPU_OPERATOR_REAL sum = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < 0
         || f >= s.nFaces
         || (f >= s.nInternalFaces && s.riemannBoundaryKind[f] == 4)
        )
        {
            continue;
        }
        const int other = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? (s.faceOwner[f] == c ? s.faceNeighbour[f] : s.faceOwner[f])
          : -1;
        const GPU_OPERATOR_REAL rhoFace = other >= 0
          ? GPU_OPERATOR_R(0.5)*(s.rho[c] + s.rho[other]) : s.rho[c];
        GPU_OPERATOR_REAL muTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL kTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL directWallHeatFlux = GPU_OPERATOR_R(0.0);
        int directWallHeatFluxActive = 0;
        const int boundaryKind = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? 0 : s.riemannBoundaryKind[f];
        gasFaceSubgridTransportProperties
        (
            s, f, c, other, boundaryKind, rhoFace,
            muTurbulent, kTurbulent,
            directWallHeatFlux, directWallHeatFluxActive
        );
        (void)directWallHeatFlux;
        (void)directWallHeatFluxActive;
                                                                         
        const GPU_OPERATOR_REAL muEffective = s.gasMu + muTurbulent;
        const GPU_OPERATOR_REAL kEffective = molecularGasConductivity(s) + kTurbulent;
        const GPU_OPERATOR_REAL rhoSafe = clampMin(rhoFace, s.rhoMin);
        const GPU_OPERATOR_REAL nu = muEffective/rhoSafe;
        const GPU_OPERATOR_REAL thermalAlpha = kEffective/(rhoSafe*s.gasCp + OfSmall);
        sum += fmax(nu, thermalAlpha)*s.magSf[f]*s.deltaCoeffs[f];
    }
    const GPU_OPERATOR_REAL d = dt*sum/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL equivalentCo =
        targetMaxCo*d/clampMin(s.maxDiffusionNumber, OfSmall);
    s.gasDiffusionNumber[c] = finiteDevice(equivalentCo)
      ? equivalentCo : OfGreat;
}

__global__ void computeSstStabilityNumberKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL diffusionRate = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < 0
         || f >= s.nFaces
         || (f >= s.nInternalFaces
          && (s.riemannBoundaryKind[f] == 3
           || s.riemannBoundaryKind[f] == 4))
        )
        {
            continue;
        }
        const int other = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? (s.faceOwner[f] == c ? s.faceNeighbour[f] : s.faceOwner[f])
          : -1;
        const GPU_OPERATOR_REAL rhoFace = other >= 0
          ? GPU_OPERATOR_R(0.5)*(s.rho[c] + s.rho[other])
          : (s.riemannBoundaryKind[f] == 2
            ? riemannFacePrimitiveForGradient(s, c, f).rho : s.rho[c]);
        const GPU_OPERATOR_REAL f1Face = other >= 0
          ? GPU_OPERATOR_R(0.5)*(s.sstF1[c] + s.sstF1[other]) : s.sstF1[c];
        const GPU_OPERATOR_REAL nu = s.gasMu/clampMin(rhoFace, s.rhoMin);
        const bool physicalWall =
            f >= s.nInternalFaces
         && !isPeriodicFace(s, f)
         && s.riemannBoundaryKind[f] == 2;
        GPU_OPERATOR_REAL nutFace = other >= 0
          ? GPU_OPERATOR_R(0.5)*(s.nut[c] + s.nut[other]) : s.nut[c];
        if (physicalWall)
        {
            nutFace = GPU_OPERATOR_R(0.0);
            if (s.sstWallTreatment == 1)
            {
                const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL dux = s.Ux[c] - wallUx;
                const GPU_OPERATOR_REAL duy = s.Uy[c] - wallUy;
                const GPU_OPERATOR_REAL duz = s.Uz[c] - wallUz;
                nutFace = ugkpwall::spaldingWallState
                (
                    sqrt(dux*dux + duy*duy + duz*duz),
                    s.sstWallDistance[c],
                    nu,
                    s.sstWallKappa,
                    s.sstWallE
                ).nut;
            }
        }
        const GPU_OPERATOR_REAL maximumDiffusivity = fmax
        (
            nu + ugkwp::sstAlphaK(f1Face, s.sstCoefficients)*nutFace,
            nu + ugkwp::sstAlphaOmega(f1Face, s.sstCoefficients)*nutFace
        );
        diffusionRate +=
            (rhoFace/clampMin(s.rho[c], s.rhoMin))
           *maximumDiffusivity*s.magSf[f]*s.deltaCoeffs[f];
    }
    const GPU_OPERATOR_REAL diffusionNumber =
        dt*diffusionRate/clampMin(s.V[c], OfSmall);

    GPU_OPERATOR_REAL divU = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL s2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gByNu = GPU_OPERATOR_R(0.0);
    sstVelocityInvariants(s, c, divU, s2, gByNu);
    const GPU_OPERATOR_REAL gradDot =
        s.gradKX[c]*s.gradOmegaX[c]
      + s.gradKY[c]*s.gradOmegaY[c]
      + s.gradKZ[c]*s.gradOmegaZ[c];
    const GPU_OPERATOR_REAL cd = ugkwp::sstCrossDiffusion
    (
        s.omega[c],
        gradDot,
        s.sstCoefficients
    );
    const GPU_OPERATOR_REAL sourceK = s.rho[c]*
    (
        sstKProductionForCell(s, c, gByNu)
      - (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    const GPU_OPERATOR_REAL sourceOmega = ugkwp::sstOmegaSource
    (
        s.rho[c], s.k[c], s.omega[c], divU, gByNu, s2,
        s.sstF1[c], s.sstF2[c], cd, s.sstCoefficients
    );
    const GPU_OPERATOR_REAL sourceNumber = fmax
    (
        fabs(dt*sourceK)/clampMin(s.rhoK[c], s.rho[c]*s.sstKMin),
        fabs(dt*sourceOmega)
       /clampMin(s.rhoOmega[c], s.rho[c]*s.sstOmegaMin)
    );
    const GPU_OPERATOR_REAL equivalentCo = targetMaxCo*fmax
    (
        diffusionNumber/clampMin(s.maxDiffusionNumber, OfSmall),
        sourceNumber/clampMin(s.sstMaxSourceNumber, OfSmall)
    );
    s.sstSourceNumber[c] = finiteDevice(equivalentCo)
      ? equivalentCo : OfGreat;
}

__global__ void applyGasFluxDivergenceByCellKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL dRho = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL dRhoUx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL dRhoUy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL dRhoUz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL dRhoE = GPU_OPERATOR_R(0.0);

    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int faceI = s.cellFaceId[start + i];
        if (faceI < 0 || faceI >= s.nFaces)
        {
            continue;
        }

        GPU_OPERATOR_REAL sign = GPU_OPERATOR_R(0.0);
        if (s.faceOwner[faceI] == c)
        {
            sign = -GPU_OPERATOR_R(1.0);
        }
        else if (s.faceNeighbour[faceI] == c)
        {
            sign = GPU_OPERATOR_R(1.0);
        }
        else
        {
            continue;
        }

        dRho += sign*s.gasPhiRho[faceI];
        dRhoUx += sign*s.gasPhiRhoUx[faceI];
        dRhoUy += sign*s.gasPhiRhoUy[faceI];
        dRhoUz += sign*s.gasPhiRhoUz[faceI];
        dRhoE += sign*s.gasPhiRhoE[faceI];
    }

    const GPU_OPERATOR_REAL scale = dt/clampMin(s.V[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoBefore = s.rho[c];
    const GPU_OPERATOR_REAL rhoAfter = rhoBefore + scale*dRho;
    if (!finiteDevice(rhoAfter) || rhoAfter <= GPU_OPERATOR_R(0.0))
    {
        asm("trap;");
    }
    s.rho[c] += scale*dRho;
    s.rhoUx[c] += scale*dRhoUx;
    s.rhoUy[c] += scale*dRhoUy;
    s.rhoUz[c] += scale*dRhoUz;
    s.rhoE[c] += scale*dRhoE;
}

__global__ void saveGasConservativeStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.rhoNext[c] = s.rho[c];
    s.rhoUxNext[c] = s.rhoUx[c];
    s.rhoUyNext[c] = s.rhoUy[c];
    s.rhoUzNext[c] = s.rhoUz[c];
    s.rhoENext[c] = s.rhoE[c];
    if (s.sstConfigured != 0)
    {
        s.rhoKInitial[c] = s.rhoK[c];
        s.rhoOmegaInitial[c] = s.rhoOmega[c];
    }
}

__global__ void blendGasConservativeStateKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_REAL initialWeight,
    const GPU_OPERATOR_REAL stageWeight
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.rho[c] =
        initialWeight*s.rhoNext[c] + stageWeight*s.rho[c];
    s.rhoUx[c] =
        initialWeight*s.rhoUxNext[c] + stageWeight*s.rhoUx[c];
    s.rhoUy[c] =
        initialWeight*s.rhoUyNext[c] + stageWeight*s.rhoUy[c];
    s.rhoUz[c] =
        initialWeight*s.rhoUzNext[c] + stageWeight*s.rhoUz[c];
    s.rhoE[c] =
        initialWeight*s.rhoENext[c] + stageWeight*s.rhoE[c];
    if (s.sstConfigured != 0)
    {
        s.rhoK[c] =
            initialWeight*s.rhoKInitial[c] + stageWeight*s.rhoK[c];
        s.rhoOmega[c] =
            initialWeight*s.rhoOmegaInitial[c]
          + stageWeight*s.rhoOmega[c];
    }
}

__device__ void recoverGasPrimitiveCell(DeviceState& s, const int c)
{
    const GPU_OPERATOR_REAL rhoSafe =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL ux = finiteOr(s.rhoUx[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uy = finiteOr(s.rhoUy[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uz = finiteOr(s.rhoUz[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL kinetic = GPU_OPERATOR_R(0.5)*rhoSafe*(ux*ux + uy*uy + uz*uz);
    const GPU_OPERATOR_REAL minimumInternalEnergy = fmax
    (
        s.rhoMin,
        rhoSafe*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - GPU_OPERATOR_R(1.0), OfSmall)
    );
    const GPU_OPERATOR_REAL internalE = clampMin
    (
        finiteOr(s.rhoE[c] - kinetic, minimumInternalEnergy),
        minimumInternalEnergy
    );
    const GPU_OPERATOR_REAL p = (s.gammaGas - GPU_OPERATOR_R(1.0))*internalE;
    const GPU_OPERATOR_REAL T = p/(rhoSafe*s.Rgas);

    s.rho[c] = rhoSafe;
    s.rhoUx[c] = rhoSafe*ux;
    s.rhoUy[c] = rhoSafe*uy;
    s.rhoUz[c] = rhoSafe*uz;
    s.rhoE[c] = kinetic + internalE;
    s.Ux[c] = ux;
    s.Uy[c] = uy;
    s.Uz[c] = uz;
    s.p[c] = p;
    s.Tgas[c] = T;
}

__global__ void recoverGasPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    recoverGasPrimitiveCell(s, c);
}
