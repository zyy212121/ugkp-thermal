#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
// Interpolate the conservative cell diffusion coefficients with the same
// owner-oriented weight used by the face flux, including coupled faces.
__device__ void sstInternalRhoDiffusivities
(
    const DeviceState& s,
    const int f,
    const int nei,
    GPU_OPERATOR_REAL& rhoDk,
    GPU_OPERATOR_REAL& rhoDomega
)
{
    const int own = s.faceOwner[f];
    const GPU_OPERATOR_REAL weight = clampRange
    (
        s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0)
    );
    const GPU_OPERATOR_REAL nuOwn = s.gasMu/clampMin(s.rho[own], s.rhoMin);
    const GPU_OPERATOR_REAL nuNei = s.gasMu/clampMin(s.rho[nei], s.rhoMin);
    rhoDk =
        weight*s.rho[own]
       *(nuOwn + ugkwp::sstAlphaK(s.sstF1[own], s.sstCoefficients)*s.nut[own])
      + (GPU_OPERATOR_R(1.0) - weight)*s.rho[nei]
       *(nuNei + ugkwp::sstAlphaK(s.sstF1[nei], s.sstCoefficients)*s.nut[nei]);
    rhoDomega =
        weight*s.rho[own]
       *(nuOwn + ugkwp::sstAlphaOmega(s.sstF1[own], s.sstCoefficients)*s.nut[own])
      + (GPU_OPERATOR_R(1.0) - weight)*s.rho[nei]
       *(nuNei + ugkwp::sstAlphaOmega(s.sstF1[nei], s.sstCoefficients)*s.nut[nei]);
}

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
            nutFace = ugkpwall::spaldingWallStateFromNormalGradient
            (
                sqrt(dux*dux + duy*duy + duz*duz),
                s.sstWallDistance[own],
                nuFace,
                sqrt(dux*dux + duy*duy + duz*duz)*s.deltaCoeffs[f],
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

    GPU_OPERATOR_REAL rhoDk = rhoFace*dk;
    GPU_OPERATOR_REAL rhoDomega = rhoFace*domega;
    if (nei >= 0)
    {
        sstInternalRhoDiffusivities(s, f, nei, rhoDk, rhoDomega);
    }

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
        massFlux*kUpwind - rhoDk*snGradK*area;
    s.sstPhiRhoOmega[f] =
        massFlux*omegaUpwind - rhoDomega*snGradOmega*area;
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
    const GPU_OPERATOR_REAL production = s.nut[c]*gByNu;
    const GPU_OPERATOR_REAL productionLimit =
        s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[c]*s.omega[c];
    if (s.sstWallTreatment != 1)
    {
        return fmin(production, productionLimit);
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
        const GPU_OPERATOR_REAL magGradU = sqrt(dux*dux + duy*duy + duz*duz)*s.deltaCoeffs[f];
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
    const GPU_OPERATOR_REAL wallBlendedProduction = wallCount > 0
      ? wallProductionSum/GPU_OPERATOR_REAL(wallCount)
      : production;
    // OF applies Pk after the wall function has updated/averaged G.
    return fmin
    (
        wallBlendedProduction,
        productionLimit
    );
}

__device__ void sstSourcesForCell
(
    const DeviceState& s,
    const int c,
    const GPU_OPERATOR_REAL divU,
    GPU_OPERATOR_REAL& sourceK,
    GPU_OPERATOR_REAL& sourceOmega
)
{
    GPU_OPERATOR_REAL traceGradU = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL s2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gByNu = GPU_OPERATOR_R(0.0);
    sstVelocityInvariants(s, c, traceGradU, s2, gByNu);
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
    sourceK = s.rho[c]*
    (
        kProduction
      - (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    sourceOmega = ugkwp::sstOmegaSource
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

    bool constrainedOmega = false;
    GPU_OPERATOR_REAL fluxK = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL fluxOmega = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL volumeFlux = GPU_OPERATOR_R(0.0);
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
        // Static mesh: absolute volumetric phi = finalized mass phi / rho_f.
        // The flux sum is outward, opposite the conservative RHS sign.
        volumeFlux -= sign*s.gasPhiRho[f]/sstFaceDensity(s, f);
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1)
         && f >= s.nInternalFaces
         && s.riemannBoundaryKind[f] == 2
        );
    }

    const GPU_OPERATOR_REAL divU = volumeFlux/clampMin(s.V[c], OfSmall);
    GPU_OPERATOR_REAL sourceK, sourceOmega;
    sstSourcesForCell(s, c, divU, sourceK, sourceOmega);
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + sourceK);
    const GPU_OPERATOR_REAL deltaRhoOmega = dt*(fluxOmega*invV + sourceOmega);
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.sstSourceNumber[c] = fmax
    (
        fabs(dt*sourceK)/clampMin(s.rhoK[c], rhoKFloor),
        constrainedOmega ? GPU_OPERATOR_R(0.0)
          : fabs(dt*sourceOmega)/clampMin(s.rhoOmega[c], rhoOmegaFloor)
    );
    s.rhoK[c] =
        clampMin(finiteOr(s.rhoK[c] + deltaRhoK, rhoKFloor), rhoKFloor);
    // Both wall treatments constrain the adjacent-cell omega equation.
    // Suppress its RHS, then primitive recovery projects to the refreshed
    // target after k and gas density evolve. Other equations remain active.
    if (!constrainedOmega)
    {
        s.rhoOmega[c] = clampMin
        (
            finiteOr(s.rhoOmega[c] + deltaRhoOmega, rhoOmegaFloor),
            rhoOmegaFloor
        );
    }
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

    // Courant-only scratch lifetime: energy-flux storage holds amaxSf until
    // computeGasConvectiveCourantByCellKernel consumes it. The next gas face
    // flux overwrites all gasPhi arrays before any gas conservative update.
    // Only SST consumes mass here; its helper initializes every return path.
    if (s.sstConfigured != 0)
    {
        GPU_OPERATOR_REAL mass, mx, my, mz, energy;
        computeRiemannGasFaceFluxDevice<false, true>
        (
            s, f, mass, mx, my, mz, energy
        );
        s.gasPhiRho[f] = mass;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        s.gasPhiRhoE[f] = OfGreat;
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
        s.gasPhiRhoE[f] = GPU_OPERATOR_R(0.0);
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
            s.gasPhiRhoE[f] = OfGreat;
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

    s.gasPhiRhoE[f] = finiteDevice(amaxSf) ? amaxSf : OfGreat;
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
        sumAmaxSf += finiteOr(s.gasPhiRhoE[f], OfGreat);
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

// Current-state timestep estimate: the predictor uses fresh pre-positivity
// Riemann mass flux, while evolution uses the finalized stage flux. Summing
// negative and positive outward volume flux separately bounds divU after any
// face positivity factor in [0,1], without cancellation between face signs.
// The shared source at divU=0 supplies the affine intercept. This fixed-state
// bound does not cover changes to gradients/F1 or later RK states.
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
    GPU_OPERATOR_REAL minimumVolumeFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL maximumVolumeFlux = GPU_OPERATOR_R(0.0);
    bool constrainedOmega = false;
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
        const GPU_OPERATOR_REAL outwardSign = s.faceOwner[f] == c
          ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        const GPU_OPERATOR_REAL outwardVolumeFlux =
            outwardSign*sstPredictorMassFlux(s, f)/sstFaceDensity(s, f);
        minimumVolumeFlux += fmin(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        maximumVolumeFlux += fmax(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1)
         && f >= s.nInternalFaces && s.riemannBoundaryKind[f] == 2
        );
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
                nutFace = ugkpwall::spaldingWallStateFromNormalGradient
                (
                    sqrt(dux*dux + duy*duy + duz*duz),
                    s.sstWallDistance[c],
                    nu,
                    sqrt(dux*dux + duy*duy + duz*duz)*s.deltaCoeffs[f],
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
        if (other >= 0)
        {
            GPU_OPERATOR_REAL rhoDk;
            GPU_OPERATOR_REAL rhoDomega;
            sstInternalRhoDiffusivities
            (
                s, f, coupledFaceNeighbour(s, f), rhoDk, rhoDomega
            );
            diffusionRate +=
                fmax(rhoDk, rhoDomega)/clampMin(s.rho[c], s.rhoMin)
               *s.magSf[f]*s.deltaCoeffs[f];
        }
        else
        {
            diffusionRate +=
                (rhoFace/clampMin(s.rho[c], s.rhoMin))
               *maximumDiffusivity*s.magSf[f]*s.deltaCoeffs[f];
        }
    }
    const GPU_OPERATOR_REAL diffusionNumber =
        dt*diffusionRate/clampMin(s.V[c], OfSmall);

    const GPU_OPERATOR_REAL invVolume = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL minimumDivU = minimumVolumeFlux*invVolume;
    const GPU_OPERATOR_REAL maximumDivU = maximumVolumeFlux*invVolume;
    GPU_OPERATOR_REAL sourceKAtZeroDiv, sourceOmegaAtZeroDiv;
    sstSourcesForCell
    (
        s, c, GPU_OPERATOR_R(0.0), sourceKAtZeroDiv, sourceOmegaAtZeroDiv
    );
    const GPU_OPERATOR_REAL compressionK =
        (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*s.rho[c]*s.k[c];
    const GPU_OPERATOR_REAL compressionOmega =
        (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*s.rho[c]
       *ugkwp::sstGamma(s.sstF1[c], s.sstCoefficients)*s.omega[c];
    const GPU_OPERATOR_REAL sourceKBound = fmax
    (
        fabs(sourceKAtZeroDiv - compressionK*minimumDivU),
        fabs(sourceKAtZeroDiv - compressionK*maximumDivU)
    );
    const GPU_OPERATOR_REAL sourceOmegaBound = fmax
    (
        fabs(sourceOmegaAtZeroDiv - compressionOmega*minimumDivU),
        fabs(sourceOmegaAtZeroDiv - compressionOmega*maximumDivU)
    );
    const GPU_OPERATOR_REAL sourceNumber = fmax
    (
        fabs(dt)*sourceKBound/clampMin(s.rhoK[c], s.rho[c]*s.sstKMin),
        constrainedOmega ? GPU_OPERATOR_R(0.0) : fabs(dt)*sourceOmegaBound
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
