#pragma once
#include <cfloat>
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
#include "gasTransport/GasBudgetAudit.H"
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
            (boundaryKind == 2 && s.sstWallTreatment != 1)
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
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerFaceSlot(s,f);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))
            {if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            s.sstPhiRhoK[f]=s.gasBoundaryLayer.exchange[slot].k;
            // The selected omega row is algebraic. No independent physical
            // wall omega injection or diffusion is imposed on top of it.
            s.sstPhiRhoOmega[f]=GPU_OPERATOR_R(0.0);
        }
    }
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
    const GPU_OPERATOR_REAL production = s.nut[c]*gByNu;
    const GPU_OPERATOR_REAL productionLimit =
        s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[c]*s.omega[c];
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0 && ugkwp::gasBoundaryLayerSlotReady(s,slot))
        {
            const auto& wall=s.gasBoundaryLayer.sst[slot];
            // OpenCFD omegaWallFunction: G=(nu+nut)*|dU/dn|*Cmu^1/4
            // *sqrt(k)/(kappa*y). Numerical traction supplies rho*(nu+nut)
            // *|dU/dn|; only the actual coarse k enters the inventory source.
            const GPU_OPERATOR_REAL wallProduction=wall.wallShearStress/s.rho[c]
                *sqrt(sqrt(s.sstWallCmu))*sqrt(s.k[c])
                /(s.sstWallKappa*wall.ownerDistance);
            return fmin(wallProduction,productionLimit);
        }
    }
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

template<class GasState>
__device__ void sstSourcesForCell
(
    const GasState& s,
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

// Selected numerical walls use a gross source activity, so production and
// destruction cannot hide stiffness by cancellation. No new constitutive pass
// is needed: P+D = (P-D-C)+C+2D, with C=(2/3)*rho*k*div(U).
template<class GasState>
__device__ GPU_OPERATOR_REAL sstKSourceActivity
(
    const GasState& s,const int c,const GPU_OPERATOR_REAL netSource,
    const GPU_OPERATOR_REAL divU
)
{
    if(ugkwp::gasBoundaryLayerOwnerSlot(s,c)<0)return fabs(netSource);
    const GPU_OPERATOR_REAL compression=GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0)
        *s.rho[c]*s.k[c]*divU;
    const GPU_OPERATOR_REAL destruction=s.sstCoefficients.betaStar*s.rho[c]
        *s.k[c]*s.omega[c];
    return netSource+compression+GPU_OPERATOR_R(2.0)*destruction+fabs(compression);
}
template<class GasState>
__device__ GPU_OPERATOR_REAL sstKSourceNumber
(
    const GasState& s,const int c,const GPU_OPERATOR_REAL activity,
    const GPU_OPERATOR_TIME dt
)
{
    const GPU_OPERATOR_REAL inventory=ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0
        ?s.rhoK[c]:clampMin(s.rhoK[c],s.rho[c]*s.sstKMin);
    if(inventory==GPU_OPERATOR_R(0.0))return activity==GPU_OPERATOR_R(0.0)
        ?GPU_OPERATOR_R(0.0):OfGreat;
    return fabs(dt)*activity/inventory;
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
    bool constrainedOmega = ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
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
        if(moving && (ugkwp::gasFaceFailure(s,f)!=0 || !finiteDevice(s.sstPhiRhoK[f]) || !finiteDevice(s.sstPhiRhoOmega[f])))
        { if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState))asm("trap;"); return; }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? -GPU_OPERATOR_R(1.0) : GPU_OPERATOR_R(1.0);
        fluxK += sign*s.sstPhiRhoK[f];
        fluxOmega += sign*s.sstPhiRhoOmega[f];
        // Convert finalized relative ALE mass flux back to absolute fluid
        // volume flux for compression; mesh motion is not a physical source.
        GPU_OPERATOR_REAL absoluteVolumeFlux=s.gasPhiRho[f]/sstFaceDensity(s, f);
        if(moving)
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
            if(!ugkwp::gasGeometryFaceFrame(s,f,dt,frame))
            {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
            absoluteVolumeFlux+=frame.meshVolumeRate;
        }
        volumeFlux -= sign*absoluteVolumeFlux;
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1 || s.sstWallTreatment == 2)
         && f >= s.nInternalFaces
         && s.riemannBoundaryKind[f] == 2
        );
    }

    const GPU_OPERATOR_REAL divU = volumeFlux/clampMin(oldVolume, OfSmall);
    GPU_OPERATOR_REAL sourceK, sourceOmega;
    sstSourcesForCell(s, c, divU, sourceK, sourceOmega);
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))
            {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            const auto& closure=s.gasBoundaryLayer.sst[slot];
            constexpr GPU_OPERATOR_REAL wallVolumeTolerance=GPU_OPERATOR_R(256.0)
                *(sizeof(GPU_OPERATOR_REAL)==sizeof(float)?FLT_EPSILON:DBL_EPSILON);
            if(!finiteDevice(closure.volume) || !(closure.volume>0)
                || !finiteDevice(closure.wallShearStress) || closure.wallShearStress<0
                || !finiteDevice(closure.ownerDistance) || !(closure.ownerDistance>0)
                || !(s.sstWallCmu>0) || !(s.sstWallKappa>0)
                || fabs(closure.volume-oldVolume)>wallVolumeTolerance*oldVolume)
            {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
            // Source remains the inventory-based SST wall-function source.
            // Never substitute the donor-matched auxiliary profile integral.
        }
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + sourceK);
    const GPU_OPERATOR_REAL deltaRhoOmega = dt*(fluxOmega*invV + sourceOmega);
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    // Compare in the source-time type: a float/double fmax otherwise selects
    // a host-only overload in CUDA. Keep dt products wide until final storage.
    s.sstSourceNumber[c] = fmax
    (
        GPU_OPERATOR_TIME(sstKSourceNumber(s,c,sstKSourceActivity(s,c,sourceK,divU),dt)),
        constrainedOmega ? GPU_OPERATOR_R(0.0)
          : fabs(dt*sourceOmega)/clampMin(s.rhoOmega[c], rhoOmegaFloor)
    );
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
        if(ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0
            && (!finiteDevice(s.sstSourceNumber[c]) || s.sstSourceNumber[c]>s.sstMaxSourceNumber))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::SourceStepLimit))asm("trap;");return;}
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
        s.gasPhiRhoE[f]=OfGreat;
        if(s.sstConfigured!=0)s.gasPhiRho[f]=GPU_OPERATOR_R(0.0);
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");
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
            (s.riemannBoundaryKind[f] == 1 && !ugkwp::gasMovingGeometry(s))
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
    const GasPrimDevice left = gasCellPrimitive(s,own);
    GasPrimDevice right = left;
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            s.gasPhiRhoE[f] = OfGreat;
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

    s.gasPhiRhoE[f] = finiteDevice(amaxSf) ? amaxSf : OfGreat;
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
        sumAmaxSf += finiteOr(s.gasPhiRhoE[f], OfGreat);
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
        // Match the ordinary face-flux channels. A selected wall returns its
        // independent physical exchange before those ordinary boundary flags;
        // retain its existing coefficient bounds even for a cold profile.
        const bool wallExchange=ugkwp::gasBoundaryLayerFaceSlot(s,f)>=0;
        if(boundaryKind==3||boundaryKind==4
            ||(boundaryKind==1&&s.riemannBoundaryTFix[f]==0))continue;

        gasFaceSubgridTransportProperties
        (
            s, f, c, other, boundaryKind, rhoFace,
            muTurbulent, kTurbulent,
            directWallHeatFlux, directWallHeatFluxActive
        );
        (void)directWallHeatFlux;
        (void)directWallHeatFluxActive;
                                                                         
        const GPU_OPERATOR_REAL muEffective = boundaryKind==1
          ? GPU_OPERATOR_R(0.0) : s.gasMu + muTurbulent;
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
                const bool speciesChannel=wallExchange||other>=0
                    ||(boundaryKind==0&&s.gasSpecies.compositionBoundaryFixed[f]!=0);
                if(speciesChannel){
                    if(s.gasSpecies.diffusivity)
                        for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                            speciesDiffusivity=fmax(speciesDiffusivity,GPU_OPERATOR_R(2.0)*s.gasSpecies.diffusivity[k]);
                    speciesDiffusivity+=GPU_OPERATOR_R(2.0)*muTurbulent/(rhoSafe*s.gasSpecies.turbulentSchmidt);
                }
            }
        const bool thermalChannel=wallExchange||other>=0||s.riemannBoundaryTFix[f]!=0;
        const GPU_OPERATOR_REAL thermalAlpha = thermalChannel
          ? kEffective/(rhoSafe*heatCapacity + OfSmall) : GPU_OPERATOR_R(0.0);
        sum += fmax(fmax(nu, thermalAlpha),speciesDiffusivity)*s.magSf[f]*s.deltaCoeffs[f];
    }
    const GPU_OPERATOR_REAL d = dt*sum/clampMin(stabilityVolume, OfSmall);
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
template<class GasState>
__global__ void computeSstStabilityNumberKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo,
    const bool requireWallProfile=true
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL stabilityVolume=s.V[c],sourceVolume=s.V[c];
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume))
        {s.sstSourceNumber[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
        stabilityVolume=fmin(oldVolume,newVolume);sourceVolume=oldVolume;
    }
    GPU_OPERATOR_REAL diffusionRate = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL minimumVolumeFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL maximumVolumeFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL meshVolumeFlux = GPU_OPERATOR_R(0.0);
    bool constrainedOmega = ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
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
        if(ugkwp::gasMovingGeometry(s))
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
            if(!ugkwp::gasGeometryFaceFrame(s,f,dt,frame))
            {s.sstSourceNumber[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
            meshVolumeFlux+=outwardSign*frame.meshVolumeRate;
        }
        minimumVolumeFlux += fmin(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        maximumVolumeFlux += fmax(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1 || s.sstWallTreatment == 2)
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
        else if(ugkwp::gasBoundaryLayerFaceSlot(s,f)<0)
        {
            // Selected high-Re wall relations impose neither k nor omega
            // diffusion at the physical wall; internal-face diffusion remains.
            diffusionRate +=
                (rhoFace/clampMin(s.rho[c], s.rhoMin))
               *maximumDiffusivity*s.magSf[f]*s.deltaCoeffs[f];
        }
    }
    const GPU_OPERATOR_REAL diffusionNumber =
        dt*diffusionRate/clampMin(stabilityVolume, OfSmall);

    const GPU_OPERATOR_REAL invVolume = GPU_OPERATOR_R(1.0)/clampMin(sourceVolume, OfSmall);
    const GPU_OPERATOR_REAL minimumDivU = (minimumVolumeFlux+meshVolumeFlux)*invVolume;
    const GPU_OPERATOR_REAL maximumDivU = (maximumVolumeFlux+meshVolumeFlux)*invVolume;
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
    GPU_OPERATOR_REAL sourceKBound = fmax
    (
        fabs(sourceKAtZeroDiv - compressionK*minimumDivU),
        fabs(sourceKAtZeroDiv - compressionK*maximumDivU)
    );
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot) && requireWallProfile)
            {s.sstSourceNumber[c]=OfGreat;if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            // Cold outer estimates use coarse SST until the authoritative
            // preflight publishes current wall stress and owner omega.
            sourceKBound=fmax(
                sstKSourceActivity(s,c,sourceKAtZeroDiv-compressionK*minimumDivU,minimumDivU),
                sstKSourceActivity(s,c,sourceKAtZeroDiv-compressionK*maximumDivU,maximumDivU));
        }
    }
    const GPU_OPERATOR_REAL sourceOmegaBound = fmax
    (
        fabs(sourceOmegaAtZeroDiv - compressionOmega*minimumDivU),
        fabs(sourceOmegaAtZeroDiv - compressionOmega*maximumDivU)
    );
    const GPU_OPERATOR_REAL sourceNumber = fmax
    (
        GPU_OPERATOR_TIME(sstKSourceNumber(s,c,sourceKBound,dt)),
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
    using Energy = ugkwp::GasEnergyScalar<GasState>;
    Energy dRhoE = Energy(0);
    constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
    // Reuse the publication scratch for face sums. Each species retains the
    // original face accumulation order; no extra face/domain pass is added.
    GPU_OPERATOR_REAL trial[Ns>0?Ns:1]{};
    GPU_OPERATOR_REAL speciesFluxMagnitude=GPU_OPERATOR_R(0.0);
    // Exact binary32/binary64 epsilon, usable in CUDA without a host constexpr call.
    constexpr GPU_OPERATOR_REAL roundoff=sizeof(GPU_OPERATOR_REAL)==sizeof(float)
        ? GPU_OPERATOR_REAL(0x1p-23) : GPU_OPERATOR_REAL(0x1p-52);
    if constexpr(Ns>0)
        if(ugkwp::mixtureGasActive(s)&&s.gasSpecies.cellStatus[c]!=0)return;

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

        if constexpr(Ns>0)
        if(ugkwp::mixtureGasActive(s))
        {
            if(s.gasSpecies.faceStatus[faceI]!=0)
            {s.gasSpecies.cellStatus[c]=s.gasSpecies.faceStatus[faceI];return;}
            GPU_OPERATOR_REAL faceSum=GPU_OPERATOR_R(0.0),faceMagnitude=GPU_OPERATOR_R(0.0);
            for(int k=0;k<Ns;++k)
            {
                const GPU_OPERATOR_REAL flux=s.gasSpecies.flux[k*s.nFaces+faceI];
                if(!finiteDevice(flux))
                {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
                trial[k]+=sign*flux;faceSum+=flux;faceMagnitude+=fabs(flux);
                if(ugkwp::gasBudgetEnabled(s) && ugkwp::gasBudgetExternalFace(s,faceI))
                    ugkwp::gasBudgetFlux(s,c,k+2,dt,sign*flux);
            }
            const GPU_OPERATOR_REAL massFlux=s.gasPhiRho[faceI];
            const GPU_OPERATOR_REAL faceBound=GPU_OPERATOR_R(8*Ns)*roundoff
                *fmax(fabs(massFlux),faceMagnitude);
            if(!finiteDevice(massFlux)||!finiteDevice(faceSum)||!finiteDevice(faceMagnitude)
                ||fabs(faceSum-massFlux)>faceBound)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            speciesFluxMagnitude+=faceMagnitude;
        }
        dRho += sign*s.gasPhiRho[faceI];
        dRhoUx += sign*s.gasPhiRhoUx[faceI];
        dRhoUy += sign*s.gasPhiRhoUy[faceI];
        dRhoUz += sign*s.gasPhiRhoUz[faceI];
        dRhoE += Energy(sign)*Energy(s.gasPhiRhoE[faceI]);
        if(ugkwp::gasBudgetEnabled(s) && ugkwp::gasBudgetExternalFace(s,faceI))
        {
            ugkwp::gasBudgetFlux(s,c,0,dt,sign*s.gasPhiRho[faceI]);
            ugkwp::gasBudgetFlux(s,c,1,dt,sign*s.gasPhiRhoE[faceI]);
        }
    }

    GPU_OPERATOR_REAL scale = dt/clampMin(s.V[c], s.rhoMin);
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            if(!ugkwp::gasFinite(s.V[c]) || !(s.V[c]>GPU_OPERATOR_R(0.0)))
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidGeometry);return;}
            scale=dt/s.V[c];
        }
    Energy energyScale = Energy(dt)/Energy(clampMin(s.V[c], s.rhoMin));
    if (ugkwp::mixtureGasActive(s)) energyScale=Energy(dt)/Energy(s.V[c]);
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value) energyScale=scale;
    Energy energyAfter = s.rhoE[c] + energyScale*dRhoE;
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
        if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
            energyAfter=s.rhoE[c]*densityRatio+scale*dRhoE;
        else energyAfter=(s.rhoE[c]*Energy(oldVolume)+Energy(dt)*dRhoE)/Energy(newVolume);
        if(!finiteDevice(dRho)||!finiteDevice(dRhoUx)||!finiteDevice(dRhoUy)||!finiteDevice(dRhoUz)||!ugkwp::gasEnergyFinite(dRhoE)
            || !finiteDevice(s.rhoUx[c]*densityRatio+scale*dRhoUx)
            || !finiteDevice(s.rhoUy[c]*densityRatio+scale*dRhoUy)
            || !finiteDevice(s.rhoUz[c]*densityRatio+scale*dRhoUz)
            || !ugkwp::gasEnergyFinite(energyAfter))
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
            for(int k=0;k<Ns;++k)
            {
                trial[k]=s.gasSpecies.rho[k*s.nCells+c]*densityRatio+scale*trial[k];
                if(!ugkwp::gasFinite(trial[k])||trial[k]<GPU_OPERATOR_R(0.0))
                {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::NegativeInventory);return;}
            }
            // Density and partial densities represent the same mass coordinate.
            // Independent rounded additions otherwise accumulate a closure drift
            // even when every face species flux sums to its total mass flux.
            // Reconstruct one populated coordinate only within the local rounding
            // bound; malformed incoming states and genuine flux defects still fail.
            GPU_OPERATOR_REAL beforeSum=GPU_OPERATOR_R(0.0),afterSum=GPU_OPERATOR_R(0.0);
            int dependent=Ns-1;
            for(int k=0;k<Ns;++k)
            {
                const GPU_OPERATOR_REAL previous=s.gasSpecies.rho[k*s.nCells+c];
                if(!finiteDevice(previous)||previous<GPU_OPERATOR_R(0.0))
                {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
                beforeSum+=previous;
                afterSum+=trial[k];
            }
            if(!(trial[dependent]>GPU_OPERATOR_R(0.0)))
                for(int k=0;k<Ns;++k)if(trial[k]>trial[dependent])dependent=k;
            // Only this step's actual arithmetic supplies repair capacity.
            // An incoming closure defect does not enlarge that allowance.
            const GPU_OPERATOR_REAL bound=(GPU_OPERATOR_R(8*Ns)+GPU_OPERATOR_R(2)*count)*roundoff
                *(fmax(rhoAfter,rhoBefore*densityRatio)+fabs(scale)*speciesFluxMagnitude);
            if(!finiteDevice(rhoBefore)||!(rhoBefore>GPU_OPERATOR_R(0.0))
                ||!finiteDevice(beforeSum)||!finiteDevice(afterSum)||!finiteDevice(bound)
                || !finiteDevice(s.gasSpecies.densityClosureTolerance)
                || s.gasSpecies.densityClosureTolerance<GPU_OPERATOR_R(0.0)
                || fabs(beforeSum-rhoBefore)>s.gasSpecies.densityClosureTolerance*rhoBefore)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            // Carry an already admissible input defect in integrated mass
            // units; repair only fresh arithmetic drift, never erase the input
            // discrepancy repeatedly or use it to enlarge the repair budget.
            const GPU_OPERATOR_REAL targetSum=rhoAfter+(beforeSum-rhoBefore)*densityRatio;
            if(!finiteDevice(targetSum))
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            if(afterSum!=targetSum&&fabs(afterSum-targetSum)<=bound)
            {
                GPU_OPERATOR_REAL independent=GPU_OPERATOR_R(0.0);
                for(int k=0;k<Ns;++k)if(k!=dependent)independent+=trial[k];
                GPU_OPERATOR_REAL replacement=targetSum-independent;
                // A tiny positive last component may lack room for the signed
                // roundoff. Preserve it and use a populated carrier instead.
                if(replacement<GPU_OPERATOR_R(0.0))
                {
                    for(int k=0;k<Ns;++k)if(trial[k]>trial[dependent])dependent=k;
                    independent=GPU_OPERATOR_R(0.0);
                    for(int k=0;k<Ns;++k)if(k!=dependent)independent+=trial[k];
                    replacement=targetSum-independent;
                }
                if(!finiteDevice(replacement) || replacement<GPU_OPERATOR_R(0.0)
                    || fabs(replacement-trial[dependent])>bound)
                {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
                trial[dependent]=replacement;
            }
            GPU_OPERATOR_REAL publishedSum=GPU_OPERATOR_R(0.0);
            for(int k=0;k<Ns;++k)publishedSum+=trial[k];
            if(!finiteDevice(publishedSum)
                ||fabs(publishedSum-rhoAfter)>s.gasSpecies.densityClosureTolerance*rhoAfter)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            for(int k=0;k<Ns;++k)s.gasSpecies.rho[k*s.nCells+c]=trial[k];
        }
    }
    if(ugkwp::gasMovingGeometry(s))
    {
        s.rho[c]=rhoAfter;s.rhoUx[c]=s.rhoUx[c]*densityRatio+scale*dRhoUx;
        s.rhoUy[c]=s.rhoUy[c]*densityRatio+scale*dRhoUy;s.rhoUz[c]=s.rhoUz[c]*densityRatio+scale*dRhoUz;
        s.rhoE[c]=energyAfter;
    }
    else
    {
    s.rho[c] += scale*dRho;
    s.rhoUx[c] += scale*dRhoUx;
    s.rhoUy[c] += scale*dRhoUy;
    s.rhoUz[c] += scale*dRhoUz;
    s.rhoE[c] = energyAfter;
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
    ugkwp::saveGasBudgetAudit(s,c);
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
    const double energyInitialWeight,
    const double energyStageWeight
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const GPU_OPERATOR_REAL initialWeight=GPU_OPERATOR_REAL(energyInitialWeight);
    const GPU_OPERATOR_REAL stageWeight=GPU_OPERATOR_REAL(energyStageWeight);
    using Energy = ugkwp::GasEnergyScalar<GasState>;
    const GPU_OPERATOR_REAL blendedDensity=initialWeight*s.rhoNext[c]+stageWeight*s.rho[c];
    constexpr int reviewNs=ugkwp::GasStateTraits<GasState>::speciesCount;
    GPU_OPERATOR_REAL blendedSpecies[reviewNs>0?reviewNs:1]{};
    if constexpr(reviewNs>0) if(ugkwp::mixtureGasActive(s))
    {
        if(s.gasSpecies.cellStatus[c]!=0)return;
        GPU_OPERATOR_REAL sumInitial=GPU_OPERATOR_R(0),sumStage=GPU_OPERATOR_R(0),sumBlend=GPU_OPERATOR_R(0);
        int dependent=reviewNs-1;
        for(int k=0;k<reviewNs;++k)
        {
            const int i=k*s.nCells+c;
            const GPU_OPERATOR_REAL a=s.gasSpecies.initial[i],b=s.gasSpecies.rho[i];
            if(!finiteDevice(a)||!finiteDevice(b)||a<GPU_OPERATOR_R(0)||b<GPU_OPERATOR_R(0))
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            sumInitial+=a;sumStage+=b;blendedSpecies[k]=initialWeight*a+stageWeight*b;sumBlend+=blendedSpecies[k];
        }
        const GPU_OPERATOR_REAL tol=s.gasSpecies.densityClosureTolerance;
        if(!finiteDevice(tol)||tol<GPU_OPERATOR_R(0)||!finiteDevice(blendedDensity)||!(blendedDensity>GPU_OPERATOR_R(0))
            ||!finiteDevice(s.rhoNext[c])||!(s.rhoNext[c]>GPU_OPERATOR_R(0))||!finiteDevice(s.rho[c])||!(s.rho[c]>GPU_OPERATOR_R(0))
            ||fabs(sumInitial-s.rhoNext[c])>tol*s.rhoNext[c]||fabs(sumStage-s.rho[c])>tol*s.rho[c])
        {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
        // Preserve the convexly transported input defects; repair only newly
        // introduced rounding in this RK publication, not physical imbalance.
        const GPU_OPERATOR_REAL target=blendedDensity+initialWeight*(sumInitial-s.rhoNext[c])+stageWeight*(sumStage-s.rho[c]);
        constexpr GPU_OPERATOR_REAL eps=sizeof(GPU_OPERATOR_REAL)==sizeof(float)?GPU_OPERATOR_REAL(0x1p-23):GPU_OPERATOR_REAL(0x1p-52);
        const GPU_OPERATOR_REAL bound=GPU_OPERATOR_R(8*reviewNs)*eps*(fabs(initialWeight*s.rhoNext[c])+fabs(stageWeight*s.rho[c]));
        if(!finiteDevice(target)||!finiteDevice(sumBlend)||fabs(sumBlend-target)>bound)
        {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
        if(sumBlend!=target)
        {
            if(!(blendedSpecies[dependent]>GPU_OPERATOR_R(0)))
                for(int k=0;k<reviewNs;++k)if(blendedSpecies[k]>blendedSpecies[dependent])dependent=k;
            GPU_OPERATOR_REAL independent=GPU_OPERATOR_R(0);for(int k=0;k<reviewNs;++k)if(k!=dependent)independent+=blendedSpecies[k];
            GPU_OPERATOR_REAL replacement=target-independent;
            if(replacement<GPU_OPERATOR_R(0))
            {for(int k=0;k<reviewNs;++k)if(blendedSpecies[k]>blendedSpecies[dependent])dependent=k;independent=GPU_OPERATOR_R(0);for(int k=0;k<reviewNs;++k)if(k!=dependent)independent+=blendedSpecies[k];replacement=target-independent;}
            if(!finiteDevice(replacement)||replacement<GPU_OPERATOR_R(0)||fabs(replacement-blendedSpecies[dependent])>bound)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
            blendedSpecies[dependent]=replacement;
        }
        GPU_OPERATOR_REAL published=GPU_OPERATOR_R(0);for(int k=0;k<reviewNs;++k)published+=blendedSpecies[k];
        if(!finiteDevice(published)||fabs(published-blendedDensity)>tol*blendedDensity)
        {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidComposition);return;}
    }
    s.rho[c]=blendedDensity;
    s.rhoUx[c] =
        initialWeight*s.rhoUxNext[c] + stageWeight*s.rhoUx[c];
    s.rhoUy[c] =
        initialWeight*s.rhoUyNext[c] + stageWeight*s.rhoUy[c];
    s.rhoUz[c] =
        initialWeight*s.rhoUzNext[c] + stageWeight*s.rhoUz[c];
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
    {
        if (!(std::is_same<Energy,double>::value && s.rhoENext[c]==s.rhoE[c]
            && initialWeight+stageWeight==GPU_OPERATOR_R(1)))
            s.rhoE[c]=initialWeight*s.rhoENext[c]+stageWeight*s.rhoE[c];
    }
    else
        s.rhoE[c]=Energy(energyInitialWeight+energyStageWeight)*s.rhoENext[c]
            +Energy(energyStageWeight)*(s.rhoE[c]-s.rhoENext[c]);
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
        ugkwp::blendGasBudgetAudit(s,c,initialWeight,stageWeight);
    else
        ugkwp::blendGasBudgetAudit(s,c,initialWeight,stageWeight,
            energyInitialWeight,energyStageWeight);
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if (ugkwp::mixtureGasActive(s))
            for (int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
            {
                const int i=k*s.nCells+c;
                s.gasSpecies.rho[i]=blendedSpecies[k];
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
            using Energy = ugkwp::GasEnergyScalar<GasState>;
            constexpr int Ns = ugkwp::GasStateTraits<GasState>::speciesCount;
            Energy partialDensity[Ns];
            Energy densitySum = Energy(0.0);
            Energy gasConstantDensity = Energy(0.0);
            const Energy density = s.rho[c];
            for (int k = 0; k < Ns; ++k)
            {
                partialDensity[k] = species.rho[k*s.nCells+c];
                if (!ugkwp::gasFinite(partialDensity[k]) || partialDensity[k] < Energy(0.0))
                {
                    species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidComposition);
                    return;
                }
                densitySum += partialDensity[k];
            }
            if (!ugkwp::gasFinite(density) || !(density > Energy(0.0))
                || !ugkwp::gasFinite(species.densityClosureTolerance)
                || species.densityClosureTolerance < Energy(0.0)
                || fabs(densitySum-density) > species.densityClosureTolerance*density)
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidComposition);
                return;
            }
            const Energy ux = Energy(s.rhoUx[c])/density;
            const Energy uy = Energy(s.rhoUy[c])/density;
            const Energy uz = Energy(s.rhoUz[c])/density;
            const Energy internalEnergy = s.rhoE[c]
                - Energy(0.5)*density*(ux*ux+uy*uy+uz*uz);
            Energy temperature = s.Tgas[c];
            const auto status = ugkwp::invertMixtureEnergy
            (
                partialDensity, internalEnergy, species.thermo,
                ugkwp::castThermoControls<Energy>(species.thermoControls), temperature
            );
            if (status != ugkwp::ThermoStatus::Success)
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidThermodynamics);
                return;
            }
            Energy cpDensity = Energy(0.0);
            for (int k = 0; k < Ns; ++k)
            {
                gasConstantDensity += partialDensity[k]*ugkwp::universalGasConstant<Energy>()
                    /species.thermo.species[k].molarMass;
                cpDensity += partialDensity[k]*ugkwp::speciesCp(k, temperature, species.thermo);
            }
            const Energy R = gasConstantDensity/density;
            const Energy cp = cpDensity/density;
            const Energy pressure = gasConstantDensity*temperature;
            const Energy a2 = cp/(cp-R)*R*temperature;
            if (!ugkwp::gasFinite(ux) || !ugkwp::gasFinite(uy) || !ugkwp::gasFinite(uz)
                || !ugkwp::gasFinite(pressure) || !(pressure > Energy(0.0))
                || !ugkwp::gasFinite(a2) || !(a2 > Energy(0.0)))
            {
                species.cellStatus[c] = int(ugkwp::GasTransportCode::InvalidThermodynamics);
                return;
            }
            if constexpr (!std::is_same<Energy,GPU_OPERATOR_REAL>::value)
            {
                // Derived caches retain Real storage. Reject conversion overflow
                // or loss of positive pressure/sound speed before any publication.
                const GPU_OPERATOR_REAL storedPressure=GPU_OPERATOR_REAL(pressure);
                const GPU_OPERATOR_REAL storedSound=GPU_OPERATOR_REAL(sqrt(a2));
                if (!ugkwp::gasFinite(GPU_OPERATOR_REAL(ux))
                    || !ugkwp::gasFinite(GPU_OPERATOR_REAL(uy))
                    || !ugkwp::gasFinite(GPU_OPERATOR_REAL(uz))
                    || !ugkwp::gasFinite(GPU_OPERATOR_REAL(temperature))
                    || !ugkwp::gasFinite(GPU_OPERATOR_REAL(cp))
                    || !ugkwp::gasFinite(GPU_OPERATOR_REAL(R))
                    || !ugkwp::gasFinite(storedPressure) || !(storedPressure>GPU_OPERATOR_R(0))
                    || !ugkwp::gasFinite(storedSound) || !(storedSound>GPU_OPERATOR_R(0)))
                {
                    species.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidThermodynamics);
                    return;
                }
            }
            s.Ux[c] = ux; s.Uy[c] = uy; s.Uz[c] = uz;
            s.p[c] = pressure; s.Tgas[c] = temperature;
            species.soundSpeed[c] = sqrt(a2);
            species.heatCapacity[c] = cp; species.gasConstant[c] = R;
            return;
        }
    }
    using Energy = ugkwp::GasEnergyScalar<GasState>;
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
    {
        // Preserve the established homogeneous legacy numerical path.
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
    const GPU_OPERATOR_REAL kinetic = ugkwp::gasLegacyKineticEnergy(rhoSafe, ux, uy, uz);
    // Retain legacy fmax handling of non-finite operands in this branch.
    const GPU_OPERATOR_REAL minimumInternalEnergy = fmax
    (
        s.rhoMin,
        ugkwp::gasLegacyThermalMinimumInternalEnergy
        (
            rhoSafe, GPU_OPERATOR_REAL(s.Rgas), GPU_OPERATOR_REAL(s.TgasMin),
            GPU_OPERATOR_REAL(s.gammaGas), GPU_OPERATOR_REAL(OfSmall)
        )
    );
    const GPU_OPERATOR_REAL internalE = clampMin
    (
        finiteOr(s.rhoE[c] - kinetic, minimumInternalEnergy),
        minimumInternalEnergy
    );
    const GPU_OPERATOR_REAL p = (s.gammaGas - GPU_OPERATOR_R(1.0))*internalE;
    const GPU_OPERATOR_REAL T = p/(rhoSafe*s.Rgas);

    const bool repairDensity=!finiteDevice(s.rho[c]) || s.rho[c]<s.rhoMin;
    s.rho[c] = rhoSafe;
    if constexpr (std::is_same<GPU_OPERATOR_REAL,double>::value)
    {
        // Valid conserved momenta are inputs, not rounded primitive caches.
        if (repairDensity || !finiteDevice(s.rhoUx[c]/rhoSafe)) s.rhoUx[c]=rhoSafe*ux;
        if (repairDensity || !finiteDevice(s.rhoUy[c]/rhoSafe)) s.rhoUy[c]=rhoSafe*uy;
        if (repairDensity || !finiteDevice(s.rhoUz[c]/rhoSafe)) s.rhoUz[c]=rhoSafe*uz;
    }
    else
    {
        s.rhoUx[c] = rhoSafe*ux;
        s.rhoUy[c] = rhoSafe*uy;
        s.rhoUz[c] = rhoSafe*uz;
    }
    // A valid double inventory is authoritative; subtract/add can lose one ULP.
    // Keep the legacy float path and every existing invalid/floor repair.
    if constexpr (std::is_same<GPU_OPERATOR_REAL,double>::value)
    {
        if (!finiteDevice(s.rhoE[c]-kinetic)
            || s.rhoE[c]-kinetic < minimumInternalEnergy)
            s.rhoE[c] = kinetic + internalE;
    }
    else s.rhoE[c] = kinetic + internalE;
    s.Ux[c] = ux;
    s.Uy[c] = uy;
    s.Uz[c] = uz;
    s.p[c] = p;
    s.Tgas[c] = T;
    }
    else
    {
    if(ugkwp::gasMovingGeometry(s))
    {
        if(ugkwp::gasCellFailure(s,c)!=0)return;
        const GPU_OPERATOR_REAL density=s.rho[c];
        const GPU_OPERATOR_REAL ux=s.rhoUx[c]/density,uy=s.rhoUy[c]/density,uz=s.rhoUz[c]/density;
        const Energy ex=Energy(s.rhoUx[c])/Energy(density), ey=Energy(s.rhoUy[c])/Energy(density), ez=Energy(s.rhoUz[c])/Energy(density);
        const Energy internal=s.rhoE[c]-Energy(0.5)*Energy(density)*(ex*ex+ey*ey+ez*ez);
        const Energy pressure=(Energy(s.gammaGas)-Energy(1))*internal;
        const Energy temperature=pressure/(Energy(density)*Energy(s.Rgas));
        const GPU_OPERATOR_REAL storedPressure=GPU_OPERATOR_REAL(pressure),storedTemperature=GPU_OPERATOR_REAL(temperature);
        if(!finiteDevice(density)||density<s.rhoMin||!finiteDevice(ux)||!finiteDevice(uy)||!finiteDevice(uz)
            ||!finiteDevice(storedPressure)||!(storedPressure>GPU_OPERATOR_R(0.0))
            ||!finiteDevice(storedTemperature)||storedTemperature<s.TgasMin)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidThermodynamics))asm("trap;");return;}
        s.Ux[c]=ux;s.Uy[c]=uy;s.Uz[c]=uz;s.p[c]=pressure;s.Tgas[c]=temperature;
        return;
    }
    const GPU_OPERATOR_REAL rhoSafe =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL ux = finiteOr(s.rhoUx[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uy = finiteOr(s.rhoUy[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uz = finiteOr(s.rhoUz[c]/rhoSafe, GPU_OPERATOR_R(0.0));
    const bool repairDensity=!finiteDevice(s.rho[c]) || s.rho[c]<s.rhoMin;
    const GPU_OPERATOR_REAL mxCandidate=repairDensity||!finiteDevice(s.rhoUx[c]/rhoSafe)?rhoSafe*ux:s.rhoUx[c],
        myCandidate=repairDensity||!finiteDevice(s.rhoUy[c]/rhoSafe)?rhoSafe*uy:s.rhoUy[c],
        mzCandidate=repairDensity||!finiteDevice(s.rhoUz[c]/rhoSafe)?rhoSafe*uz:s.rhoUz[c];
    // Derive energy from the actual conserved candidate, never from rounded U.
    const Energy ex=Energy(mxCandidate)/Energy(rhoSafe),
        ey=Energy(myCandidate)/Energy(rhoSafe),ez=Energy(mzCandidate)/Energy(rhoSafe);
    const Energy kinetic = ugkwp::gasLegacyKineticEnergy
    (
        Energy(rhoSafe), ugkwp::gasEnergyFiniteOr(ex,Energy(0)),
        ugkwp::gasEnergyFiniteOr(ey,Energy(0)), ugkwp::gasEnergyFiniteOr(ez,Energy(0))
    );
    const Energy minimumInternalEnergy = ugkwp::gasLegacyMinimumInternalEnergy
    (
        Energy(s.rhoMin), Energy(rhoSafe), Energy(s.Rgas), Energy(s.TgasMin),
        Energy(s.gammaGas), Energy(OfSmall)
    );
    const Energy rawInternal = s.rhoE[c]-kinetic;
    const Energy internalE = ugkwp::gasEnergyClampMin
    (
        ugkwp::gasEnergyFiniteOr(rawInternal, minimumInternalEnergy),
        minimumInternalEnergy
    );
    const Energy p = (Energy(s.gammaGas)-Energy(1))*internalE;
    const Energy T = p/(Energy(rhoSafe)*Energy(s.Rgas));
    const GPU_OPERATOR_REAL storedPressure=GPU_OPERATOR_REAL(p),storedTemperature=GPU_OPERATOR_REAL(T);
    if (!finiteDevice(storedPressure) || !(storedPressure>GPU_OPERATOR_R(0))
        || !finiteDevice(storedTemperature) || storedTemperature<s.TgasMin)
    {
        // Finite authority whose derived cache cannot be represented must not
        // be silently replaced by a lower-energy float repair.
        if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidThermodynamics))asm("trap;");
        return;
    }

    s.rho[c] = rhoSafe;
    s.rhoUx[c] = mxCandidate;
    s.rhoUy[c] = myCandidate;
    s.rhoUz[c] = mzCandidate;
    // Preserve valid authoritative bits; only the existing invalid/floor repair writes.
    if (!ugkwp::gasEnergyFinite(rawInternal) || rawInternal < minimumInternalEnergy)
        s.rhoE[c] = kinetic + internalE;
    s.Ux[c] = ux;
    s.Uy[c] = uy;
    s.Uz[c] = uz;
    s.p[c] = p;
    s.Tgas[c] = T;
    }
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
