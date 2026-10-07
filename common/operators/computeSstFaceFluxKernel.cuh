#pragma once
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
template<class GasState>
__global__ void prepareGasSstAuditKernel(GasState* sp)
{
    GasState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells || !ugkwp::gasSstAuditEnabled(s))return;
    if(!ugkwp::gasSstAuditComplete(s))
    {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    if constexpr(ugkwp::GasSstAuditCapability<GasState>::value)
    {
        GPU_OPERATOR_REAL volume=s.V[c];
        if constexpr(ugkwp::GasGeometryCapability<GasState>::value)
            if(ugkwp::gasMovingGeometry(s))
            {
                if(!s.gasGeometry.oldVolume)
                {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
                volume=s.gasGeometry.oldVolume[c];
            }
        if(!finiteDevice(volume)||!(volume>GPU_OPERATOR_R(0.0)))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        s.gasSstAudit.volume[c]=volume;
    }
}

// One thread per cell validates its stage metrics and all incident faces.
// The host validates statuses before any stage can consume moving geometry.
template<class GasState>
__global__ void validateGasStageGeometryKernel(GasState* sp,const GPU_OPERATOR_TIME dt)
{
    GasState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells || !ugkwp::gasMovingGeometry(s))return;
    GPU_OPERATOR_REAL oldVolume,newVolume;
    if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume)
        || !finiteDevice(s.V[c]) || !(s.V[c]>GPU_OPERATOR_R(0.0)))
    {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
    const int start=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
    for(int i=0;i<count;++i)
    {
        const int f=s.cellFaceId[start+i];ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
        if(!ugkwp::gasGeometryFaceFrame(s,f,dt,frame))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
    }
}

// One operator implementation; scalar/time adapters are compile-time only.
// Interpolate the conservative cell diffusion coefficients with the same
// owner-oriented weight used by the face flux, including coupled faces.
template<class GasState>
__device__ void sstInternalRhoDiffusivities
(
    const GasState& s,
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

template<class GasState>
__global__ void computeSstFaceFluxKernel(GasState* sp)
{
    GasState& s = *sp;
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

template<class GasState>
__global__ void enforcePeriodicSstFluxAntisymmetryKernel(GasState* sp)
{
    GasState& s = *sp;
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

template<class GasState>
__device__ GPU_OPERATOR_REAL sstKProductionForCell
(
    const GasState& s,
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

template<class GasState>
__global__ void applySstFluxAndSourceKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    const GPU_OPERATOR_REAL beforeK=s.rhoK[c],beforeOmega=s.rhoOmega[c];
    GPU_OPERATOR_REAL oldVolume=s.V[c],newVolume=s.V[c];
    const bool moving=ugkwp::gasMovingGeometry(s);
    if(moving && !ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
    { if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;"); return; }
    if(moving && ugkwp::gasCellFailure(s,c)!=0)return;
    bool constrainedOmega = false;
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
        if(moving && (ugkwp::gasFaceFailure(s,f)!=0 || !finiteDevice(s.sstPhiRhoK[f]) || !finiteDevice(s.sstPhiRhoOmega[f])))
        { if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState))asm("trap;"); return; }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? -GPU_OPERATOR_R(1.0) : GPU_OPERATOR_R(1.0);
        fluxK += sign*s.sstPhiRhoK[f];
        fluxOmega += sign*s.sstPhiRhoOmega[f];
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1)
         && f >= s.nInternalFaces
         && s.riemannBoundaryKind[f] == 2
        );
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
        constrainedOmega ? GPU_OPERATOR_R(0.0)
          : fabs(dt*sourceOmega)/clampMin(s.rhoOmega[c], rhoOmegaFloor)
    );
    // Explicit sources are evaluated from the old density and old inventory.
    // Faces already carry the common ALE mass flux, so add no mesh term here.
    if(moving)
    {
        const GPU_OPERATOR_REAL nextK=(s.rhoK[c]*oldVolume+dt*(fluxK+oldVolume*sourceK))/newVolume;
        const GPU_OPERATOR_REAL nextOmega=(s.rhoOmega[c]*oldVolume
            +(constrainedOmega?GPU_OPERATOR_R(0.0):dt*(fluxOmega+oldVolume*sourceOmega)))/newVolume;
        if(!finiteDevice(nextK)||!finiteDevice(nextOmega))
        { if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState))asm("trap;"); return; }
        s.rhoK[c]=clampMin(nextK,rhoKFloor*oldVolume/newVolume);
        s.rhoOmega[c]=clampMin(nextOmega,rhoOmegaFloor*oldVolume/newVolume);
        ugkwp::gasSstAuditEuler(s,c,beforeK,beforeOmega,oldVolume,newVolume,dt,fluxK,fluxOmega,sourceK,sourceOmega);
        return;
    }
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
    ugkwp::gasSstAuditEuler(s,c,beforeK,beforeOmega,oldVolume,newVolume,dt,fluxK,fluxOmega,sourceK,sourceOmega);
}
template<class GasState>
__global__ void computeGasCourantFieldKernel(GasState* sp, const GPU_OPERATOR_TIME dt)
{
    GasState& s = *sp;
    (void)dt;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> meshFrame;
    if(ugkwp::gasMovingGeometry(s) && !ugkwp::gasGeometryFaceFrame(s,f,dt,meshFrame))
    {
        s.gasPhiRho[f]=OfGreat;
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");
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
            (s.riemannBoundaryKind[f] == 1 && !ugkwp::gasMovingGeometry(s))
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
    const GasPrimDevice left = gasCellPrimitive(s,own);
    GasPrimDevice right = left;
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            s.gasPhiRho[f] = OfGreat;
            return;
        }
        right = gasCellPrimitive(s,nei);
    }
    else if
    (
        s.riemannBoundaryKind[f] != 1
     && s.riemannBoundaryKind[f] != 2
    )
    {
        right = riemannBoundaryState(s, f, left);
    }
    GPU_OPERATOR_REAL unLeft = left.ux*nx + left.uy*ny + left.uz*nz;
    GPU_OPERATOR_REAL unRight = right.ux*nx + right.uy*ny + right.uz*nz;
    if constexpr (ugkwp::GasGeometryCapability<GasState>::value)
        if(ugkwp::gasMovingGeometry(s))
        {
            const GPU_OPERATOR_REAL meshNormal=meshFrame.meshVolumeRate/meshFrame.area;
            unLeft-=meshNormal;unRight-=meshNormal;
        }
    GPU_OPERATOR_REAL aLeft = sqrt(clampMin(s.gammaGas*left.p/left.rho, OfSmall));
    GPU_OPERATOR_REAL aRight = sqrt(clampMin(s.gammaGas*right.p/right.rho, OfSmall));
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            aLeft=s.gasSpecies.soundSpeed[own];
            const int other=coupledFaceNeighbour(s,f);
            aRight=s.gasSpecies.soundSpeed[other>=0?other:own];
            if(other<0 && s.riemannBoundaryKind[f]!=1 && s.riemannBoundaryKind[f]!=2)
            {
                constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
                GPU_OPERATOR_REAL y[Ns];
                for(int k=0;k<Ns;++k)y[k]=s.gasSpecies.compositionBoundaryFixed[f]
                    ?s.gasSpecies.boundaryMassFraction[k*s.nFaces+f]:s.gasSpecies.rho[k*s.nCells+own]/s.rho[own];
                const GPU_OPERATOR_REAL R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
                const GPU_OPERATOR_REAL cv=ugkwp::mixtureHeatCapacity(y,right.T,s.gasSpecies.thermo);
                aRight=sqrt((cv+R)/cv*R*right.T);
            }
        }
    const GPU_OPERATOR_REAL spectralRadius = fmax
    (
        fabs(unLeft) + aLeft,
        fabs(unRight) + aRight
    );
    const GPU_OPERATOR_REAL amaxSf = spectralRadius*area;

    s.gasPhiRho[f] = finiteDevice(amaxSf) ? amaxSf : OfGreat;
}

template<class GasState>
__global__ void computeGasConvectiveCourantByCellKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL stabilityVolume=s.V[c];
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
        {s.gasFluxPositivityScale[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        stabilityVolume=fmin(oldVolume,newVolume);
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
        GPU_OPERATOR_R(0.5)*dt*sumAmaxSf/clampMin(stabilityVolume, OfSmall);
    s.gasFluxPositivityScale[c] = finiteDevice(co) ? co : OfGreat;
}

template<class GasState>
__global__ void computeGasDiffusionNumberKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    GPU_OPERATOR_REAL stabilityVolume=s.V[c];
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
        {s.gasDiffusionNumber[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        stabilityVolume=fmin(oldVolume,newVolume);
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
        GPU_OPERATOR_REAL kEffective = molecularGasConductivity(s) + kTurbulent;
        const GPU_OPERATOR_REAL rhoSafe = clampMin(rhoFace, s.rhoMin);
        const GPU_OPERATOR_REAL nu = muEffective/rhoSafe;
        GPU_OPERATOR_REAL heatCapacity=s.gasCp, speciesDiffusivity=GPU_OPERATOR_R(0.0);
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
            if(ugkwp::mixtureGasActive(s))
            {
                const int n=other>=0?other:c;
                const GPU_OPERATOR_REAL cp=GPU_OPERATOR_R(0.5)*(s.gasSpecies.heatCapacity[c]+s.gasSpecies.heatCapacity[n]);
                const GPU_OPERATOR_REAL R=GPU_OPERATOR_R(0.5)*(s.gasSpecies.gasConstant[c]+s.gasSpecies.gasConstant[n]);
                heatCapacity=cp-R;
                if(!ugkwp::gasHasDirectConductivity(s))kEffective=s.gasMu*cp/s.gasPrClamped+kTurbulent;
                if(s.gasSpecies.diffusivity)
                    for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                        speciesDiffusivity=fmax(speciesDiffusivity,GPU_OPERATOR_R(2.0)*s.gasSpecies.diffusivity[k]);
                speciesDiffusivity+=GPU_OPERATOR_R(2.0)*muTurbulent/(rhoSafe*s.gasSpecies.turbulentSchmidt);
            }
        const GPU_OPERATOR_REAL thermalAlpha = kEffective/(rhoSafe*heatCapacity + OfSmall);
        sum += fmax(fmax(nu, thermalAlpha),speciesDiffusivity)*s.magSf[f]*s.deltaCoeffs[f];
    }
    const GPU_OPERATOR_REAL d = dt*sum/clampMin(stabilityVolume, OfSmall);
    const GPU_OPERATOR_REAL equivalentCo =
        targetMaxCo*d/clampMin(s.maxDiffusionNumber, OfSmall);
    s.gasDiffusionNumber[c] = finiteDevice(equivalentCo)
      ? equivalentCo : OfGreat;
}

template<class GasState>
__global__ void computeSstStabilityNumberKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL stabilityVolume=s.V[c];
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
        {s.sstSourceNumber[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        stabilityVolume=fmin(oldVolume,newVolume);
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
        dt*diffusionRate/clampMin(stabilityVolume, OfSmall);

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

template<class GasState>
__global__ void applyGasFluxDivergenceByCellKernel(GasState* sp, const GPU_OPERATOR_TIME dt)
{
    GasState& s = *sp;
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

    GPU_OPERATOR_REAL scale = dt/clampMin(s.V[c], s.rhoMin);
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            if(!ugkwp::gasFinite(s.V[c]) || !(s.V[c]>GPU_OPERATOR_R(0.0)))
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidGeometry);return;}
            scale=dt/s.V[c];
        }
    GPU_OPERATOR_REAL densityRatio=GPU_OPERATOR_R(1.0);
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        if(ugkwp::gasCellFailure(s,c)!=0)return;
        for(int i=0;i<count;++i)
        {
            const int f=s.cellFaceId[start+i];
            if(ugkwp::gasFaceFailure(s,f)!=0)
            {if(!ugkwp::gasRecordCellFailure(s,c,static_cast<ugkwp::GasTransportCode>(ugkwp::gasFaceFailure(s,f))))asm("trap;");return;}
        }
        densityRatio=oldVolume/newVolume;scale=dt/newVolume;
        if(!finiteDevice(dRho)||!finiteDevice(dRhoUx)||!finiteDevice(dRhoUy)||!finiteDevice(dRhoUz)||!finiteDevice(dRhoE)
            || !finiteDevice(s.rhoUx[c]*densityRatio+scale*dRhoUx)
            || !finiteDevice(s.rhoUy[c]*densityRatio+scale*dRhoUy)
            || !finiteDevice(s.rhoUz[c]*densityRatio+scale*dRhoUz)
            || !finiteDevice(s.rhoE[c]*densityRatio+scale*dRhoE))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState))asm("trap;");return;}
    }
    const GPU_OPERATOR_REAL rhoBefore = s.rho[c];
    const GPU_OPERATOR_REAL rhoAfter = rhoBefore*densityRatio + scale*dRho;
    if (!finiteDevice(rhoAfter) || rhoAfter <= GPU_OPERATOR_R(0.0))
    {
        if(ugkwp::mixtureGasActive(s)||ugkwp::gasMovingGeometry(s))
            if(ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NegativeInventory))return;
        asm("trap;");
    }
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            GPU_OPERATOR_REAL trial[Ns];
            if (s.gasSpecies.cellStatus[c]!=0) return;
            for(int k=0;k<Ns;++k)
            {
                GPU_OPERATOR_REAL change=GPU_OPERATOR_R(0.0);
                for(int i=0;i<count;++i)
                {
                    const int f=s.cellFaceId[start+i];
                    if(f<0||f>=s.nFaces)continue;
                    if (s.gasSpecies.faceStatus[f]!=0)
                    { s.gasSpecies.cellStatus[c]=s.gasSpecies.faceStatus[f]; return; }
                    const GPU_OPERATOR_REAL sign=s.faceOwner[f]==c?-GPU_OPERATOR_R(1.0):s.faceNeighbour[f]==c?GPU_OPERATOR_R(1.0):GPU_OPERATOR_R(0.0);
                    change+=sign*s.gasSpecies.flux[k*s.nFaces+f];
                }
                trial[k]=s.gasSpecies.rho[k*s.nCells+c]*densityRatio+scale*change;
                if(!ugkwp::gasFinite(trial[k])||trial[k]<GPU_OPERATOR_R(0.0))
                { s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::NegativeInventory);return; }
            }
            for(int k=0;k<Ns;++k)s.gasSpecies.rho[k*s.nCells+c]=trial[k];
        }
    }
    if(ugkwp::gasMovingGeometry(s))
    {
        s.rho[c]=rhoAfter;s.rhoUx[c]=s.rhoUx[c]*densityRatio+scale*dRhoUx;
        s.rhoUy[c]=s.rhoUy[c]*densityRatio+scale*dRhoUy;s.rhoUz[c]=s.rhoUz[c]*densityRatio+scale*dRhoUz;
        s.rhoE[c]=s.rhoE[c]*densityRatio+scale*dRhoE;
    }
    else
    {
    s.rho[c] += scale*dRho;
    s.rhoUx[c] += scale*dRhoUx;
    s.rhoUy[c] += scale*dRhoUy;
    s.rhoUz[c] += scale*dRhoUz;
    s.rhoE[c] += scale*dRhoE;
    }
}

template<class GasState>
__global__ void saveGasConservativeStateKernel(GasState* sp)
{
    GasState& s = *sp;
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
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if (ugkwp::mixtureGasActive(s))
            for (int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                s.gasSpecies.initial[k*s.nCells+c]=s.gasSpecies.rho[k*s.nCells+c];
    if (s.sstConfigured != 0)
    {
        s.rhoKInitial[c] = s.rhoK[c];
        s.rhoOmegaInitial[c] = s.rhoOmega[c];
        ugkwp::saveGasSstAudit(s,c);
    }
}

template<class GasState>
__global__ void blendGasConservativeStateKernel
(
    GasState* sp,
    const GPU_OPERATOR_REAL initialWeight,
    const GPU_OPERATOR_REAL stageWeight
)
{
    GasState& s = *sp;
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
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if (ugkwp::mixtureGasActive(s))
            for (int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
            {
                const int i=k*s.nCells+c;
                s.gasSpecies.rho[i]=initialWeight*s.gasSpecies.initial[i]+stageWeight*s.gasSpecies.rho[i];
            }
    if (s.sstConfigured != 0)
    {
        s.rhoK[c] =
            initialWeight*s.rhoKInitial[c] + stageWeight*s.rhoK[c];
        s.rhoOmega[c] =
            initialWeight*s.rhoOmegaInitial[c]
          + stageWeight*s.rhoOmega[c];
        ugkwp::blendGasSstAudit(s,c,initialWeight,stageWeight);
    }
}

template<class GasState>
__device__ void recoverGasPrimitiveCell(GasState& s, const int c)
{
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        auto& species = s.gasSpecies;
        if (species.mode != ugkwp::GasMode::SingleLegacy)
        {
            // Recovery is read-only on all accepted/conserved inventories. A
            // failed candidate leaves every primitive output unchanged.
            if (!species.cellStatus) { asm("trap;"); return; }
            if (species.cellStatus[c] != 0) return;
            if (!species.rho || !species.soundSpeed || !species.heatCapacity
                || !species.gasConstant)
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidStorage);
                return;
            }
            constexpr int Ns = ugkwp::GasStateTraits<GasState>::speciesCount;
            GPU_OPERATOR_REAL partialDensity[Ns];
            GPU_OPERATOR_REAL densitySum = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL gasConstantDensity = GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL density = s.rho[c];
            for (int k = 0; k < Ns; ++k)
            {
                partialDensity[k] = species.rho[k*s.nCells+c];
                if (!ugkwp::gasFinite(partialDensity[k]) || partialDensity[k] < GPU_OPERATOR_R(0.0))
                {
                    species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidComposition);
                    return;
                }
                densitySum += partialDensity[k];
            }
            if (!ugkwp::gasFinite(density) || !(density > GPU_OPERATOR_R(0.0))
                || !ugkwp::gasFinite(species.densityClosureTolerance)
                || species.densityClosureTolerance < GPU_OPERATOR_R(0.0)
                || fabs(densitySum-density) > species.densityClosureTolerance*density)
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidComposition);
                return;
            }
            const GPU_OPERATOR_REAL ux = s.rhoUx[c]/density;
            const GPU_OPERATOR_REAL uy = s.rhoUy[c]/density;
            const GPU_OPERATOR_REAL uz = s.rhoUz[c]/density;
            const GPU_OPERATOR_REAL internalEnergy = s.rhoE[c]
                - GPU_OPERATOR_R(0.5)*density*(ux*ux+uy*uy+uz*uz);
            GPU_OPERATOR_REAL temperature = s.Tgas[c];
            const auto status = ugkwp::invertMixtureEnergy
            (
                partialDensity, internalEnergy, species.thermo,
                species.thermoControls, temperature
            );
            if (status != ugkwp::ThermoStatus::Success)
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidThermodynamics);
                return;
            }
            GPU_OPERATOR_REAL cpDensity = GPU_OPERATOR_R(0.0);
            for (int k = 0; k < Ns; ++k)
            {
                gasConstantDensity += partialDensity[k]*ugkwp::universalGasConstant<GPU_OPERATOR_REAL>()
                    /species.thermo.species[k].molarMass;
                cpDensity += partialDensity[k]*ugkwp::speciesCp(k, temperature, species.thermo);
            }
            const GPU_OPERATOR_REAL R = gasConstantDensity/density;
            const GPU_OPERATOR_REAL cp = cpDensity/density;
            const GPU_OPERATOR_REAL pressure = gasConstantDensity*temperature;
            const GPU_OPERATOR_REAL a2 = cp/(cp-R)*R*temperature;
            if (!ugkwp::gasFinite(ux) || !ugkwp::gasFinite(uy) || !ugkwp::gasFinite(uz)
                || !ugkwp::gasFinite(pressure) || !(pressure > GPU_OPERATOR_R(0.0))
                || !ugkwp::gasFinite(a2) || !(a2 > GPU_OPERATOR_R(0.0)))
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidThermodynamics);
                return;
            }
            s.Ux[c] = ux; s.Uy[c] = uy; s.Uz[c] = uz;
            s.p[c] = pressure; s.Tgas[c] = temperature;
            species.soundSpeed[c] = sqrt(a2);
            species.heatCapacity[c] = cp; species.gasConstant[c] = R;
            return;
        }
    }
    if(ugkwp::gasMovingGeometry(s))
    {
        if(ugkwp::gasCellFailure(s,c)!=0)return;
        const GPU_OPERATOR_REAL density=s.rho[c];
        const GPU_OPERATOR_REAL ux=s.rhoUx[c]/density,uy=s.rhoUy[c]/density,uz=s.rhoUz[c]/density;
        const GPU_OPERATOR_REAL internal=s.rhoE[c]-GPU_OPERATOR_R(0.5)*density*(ux*ux+uy*uy+uz*uz);
        const GPU_OPERATOR_REAL pressure=(s.gammaGas-GPU_OPERATOR_R(1.0))*internal;
        const GPU_OPERATOR_REAL temperature=pressure/(density*s.Rgas);
        if(!finiteDevice(density)||density<s.rhoMin||!finiteDevice(ux)||!finiteDevice(uy)||!finiteDevice(uz)
            ||!finiteDevice(pressure)||!(pressure>GPU_OPERATOR_R(0.0))||!finiteDevice(temperature)||temperature<s.TgasMin)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidThermodynamics))asm("trap;");return;}
        s.Ux[c]=ux;s.Uy[c]=uy;s.Uz[c]=uz;s.p[c]=pressure;s.Tgas[c]=temperature;
        return;
    }
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

template<class GasState>
__global__ void recoverGasPrimitivesKernel(GasState* sp)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    recoverGasPrimitiveCell(s, c);
}
