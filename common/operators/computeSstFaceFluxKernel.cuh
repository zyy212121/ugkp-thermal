#pragma once
#include <cfloat>
#include "gasStencilDiagnostics.cuh"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasConservativeWallAccess.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
// Interpolate the conservative cell diffusion coefficients with the same
// owner-oriented weight used by the face flux, including coupled faces.
// Producer classification: physical scalar faults and finite residuals are
// separate, so legacy status enums never decide diagnostic severity.
template<class GasState,class Scalar>
__device__ void gasCellPhysicalDiagnostic(const GasState& s,int c,unsigned op,
    ugkp::diagnostics::Code code,Scalar value,int component=-1,
    unsigned quantity=ugkp::diagnostics::Quantity::Unspecified)
{
    using namespace ugkp::diagnostics;
    const auto record=makeRecord(code,2620,op,0,
        EntityKind::Cell,c,component,double(value),0,0,quantity);
    if(s.diagnostics)recordFatal(s.diagnostics,record);
    if(isWarning(code))return;
    // Preserve the coarse channel even when the detailed context is optional.
    // Record detail first so transport propagation cannot replace its provenance.
    const bool recorded=ugkwp::gasRecordCellFailure(s,c,code==Code::NonFinite
        ? ugkwp::GasTransportCode::NonFiniteState
        : ugkwp::GasTransportCode::InvalidThermodynamics);
    if(!recorded && !s.diagnostics)recordFatal(nullptr,record);
}
template<class GasState,class Scalar>
__device__ void gasCellResidualDiagnostic(const GasState& s,int c,unsigned op,
    Scalar observed,Scalar reference)
{
    using namespace ugkp::diagnostics;
    recordWarning(s.diagnostics,makeRecord(Code::ConservationResidual,2621,op,0,
        EntityKind::Cell,c,-1,double(observed),double(reference),double(observed),Quantity::Conservation));
}
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

    if constexpr(ugkwp::GasConservativeWallCapability<GasState>::value){
        const auto&d=s.gasBoundaryLayerModel.conservative;
        if(d.enabled&&d.faceSlot&&d.faceSlot[f]>=0&&d.active[d.faceSlot[f]]&&ugkwp::gasBoundaryLayerSlotReady(s,d.faceSlot[f])){
            const auto&v=d.face[3*d.faceSlot[f]+d.faceIndex[f]];
            s.sstPhiRhoK[f]=s.gasPhiRho[f]*v.kUpwind+v.orientation*v.diffusion[5];
            s.sstPhiRhoOmega[f]=s.gasPhiRho[f]*v.omegaUpwind+v.orientation*v.diffusion[6];
            return;
        }
    }
    if(!gasCoupledFaceStorageAvailable(s,f))
    {ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return;}
    if(f>=s.nInternalFaces && !isPeriodicFace(s,f))
    {
        const int kind=s.riemannBoundaryKind[f];
        if(kind==1 || kind==3 || kind==4)
        {s.sstPhiRhoK[f]=GPU_OPERATOR_R(0.0);s.sstPhiRhoOmega[f]=GPU_OPERATOR_R(0.0);return;}
    }
    if(!s.faceOwner)
    {ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return;}
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        gasStencilIndexFailure(s,own,own,30023);
        return;
    }
    if(!gasStencilFaceValid(s,own,f))return;
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
              ? s.riemannBoundaryUx[f] : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
              ? s.riemannBoundaryUy[f] : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
              ? s.riemannBoundaryUz[f] : GPU_OPERATOR_R(0.0);
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
                s.sstWallE, s.diagnostics, f
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
        // Correct a shared internal face once. The BVP supplies total k
        // transport for its prescribed wall-normal mass. Tangential diffusion
        // and the remaining ALE advection are retained by both adjacent cells.
        if(nei>=0&&ugkwp::gasBoundaryLayerEnabled(s)
            &&(ugkwp::gasBoundaryLayerOwnerSlot(s,own)>=0||ugkwp::gasBoundaryLayerOwnerSlot(s,nei)>=0)){
            const GPU_OPERATOR_REAL unit[3]={s.Sfx[f]/area,s.Sfy[f]/area,s.Sfz[f]/area};
            GPU_OPERATOR_REAL grad[3]={ownerWeight*s.gradKX[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradKX[nei],
                ownerWeight*s.gradKY[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradKY[nei],
                ownerWeight*s.gradKZ[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradKZ[nei]};
            GPU_OPERATOR_REAL projection=0;for(int d=0;d<3;++d)projection+=grad[d]*unit[d];
            for(int d=0;d<3;++d)grad[d]+=(snGradK-projection)*unit[d];
            GPU_OPERATOR_REAL gradOmega[3]={ownerWeight*s.gradOmegaX[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradOmegaX[nei],
                ownerWeight*s.gradOmegaY[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradOmegaY[nei],
                ownerWeight*s.gradOmegaZ[own]+(GPU_OPERATOR_R(1)-ownerWeight)*s.gradOmegaZ[nei]};
            projection=0;for(int d=0;d<3;++d)projection+=gradOmega[d]*unit[d];
            for(int d=0;d<3;++d)gradOmega[d]+=(snGradOmega-projection)*unit[d];
            GPU_OPERATOR_REAL correction=0,omegaCorrection=0;int contributors=0;
            for(int side=0;side<2;++side){const int cell=side==0?own:nei;
                const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,cell);
                if(slot<0||!ugkwp::gasBoundaryLayerSlotReady(s,slot))continue;
                const auto& closure=s.gasBoundaryLayer.sst[slot];if(closure.kFluxCount<=0)continue;
                const auto* n=closure.normal;
                const GPU_OPERATOR_REAL alignment=unit[0]*n[0]+unit[1]*n[1]+unit[2]*n[2];
                const GPU_OPERATOR_REAL cx=side==0?s.Cx[own]:sstMappedNeiCx,
                    cy=side==0?s.Cy[own]:sstMappedNeiCy,cz=side==0?s.Cz[own]:sstMappedNeiCz;
                const GPU_OPERATOR_REAL y=closure.ownerDistance+(s.faceCx[f]-cx)*n[0]+(s.faceCy[f]-cy)*n[1]+(s.faceCz[f]-cz)*n[2];
                const GPU_OPERATOR_REAL normalGrad=grad[0]*n[0]+grad[1]*n[1]+grad[2]*n[2];
                correction+=(ugkwp::boundaryLayerSstFlux(closure,y)+rhoDk*normalGrad
                    -closure.normalMassFlux*kUpwind)*alignment*area;
                const GPU_OPERATOR_REAL normalOmegaGrad=gradOmega[0]*n[0]+gradOmega[1]*n[1]+gradOmega[2]*n[2];
                omegaCorrection+=(ugkwp::boundaryLayerSstFlux(closure,y,true)+rhoDomega*normalOmegaGrad)*alignment*area;
                ++contributors;
            }
            if(contributors){s.sstPhiRhoK[f]+=correction/GPU_OPERATOR_REAL(contributors);
                s.sstPhiRhoOmega[f]+=omegaCorrection/GPU_OPERATOR_REAL(contributors);}
        }
    }
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerFaceSlot(s,f);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))
            {if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            s.sstPhiRhoK[f]=ugkwp::gasBoundaryLayerWallKFlux(s,slot,s.gasPhiRho[f],s.k[own]);
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
            // The volume closure is applied in sstSourcesForCell. The old
            // shear*sqrt(k_macro)/(kappa*y) relation belongs to a different
            // wall function and must not be superimposed on this SST BVP.
            const auto& closure=s.gasBoundaryLayer.sst[slot];
            if(closure.kFluxCount>0)return fmin(production,productionLimit);
            // Boundary-exchange coupling: use the solved wall traction and
            // the current macro k in the wall production relation. Apply the
            // macro SST limiter before adding its destruction/compression.
            // No BVP integral is rescaled to the macro inventory. Analytic
            // and material-interface adapters retain this existing relation.
            const GPU_OPERATOR_REAL wallProduction=closure.wallShearStress/s.rho[c]
                *sqrt(sqrt(s.sstWallCmu))*sqrt(s.k[c])/(s.sstWallKappa*closure.ownerDistance);
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
    if(!gasStencilRangeValid(s,c,start,count))return GPU_OPERATOR_REAL(NAN);
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if(!gasStencilFaceIndexValid(s,c,f))return GPU_OPERATOR_REAL(NAN);
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
          ? s.riemannBoundaryUx[f] : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
          ? s.riemannBoundaryUy[f] : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
          ? s.riemannBoundaryUz[f] : GPU_OPERATOR_R(0.0);
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
            production, s.diagnostics, f
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
__device__ bool sstSourcesForCell
(
    const GasState& s,
    const int c,
    const GPU_OPERATOR_REAL divU,
    GPU_OPERATOR_REAL& sourceK,
    GPU_OPERATOR_REAL& sourceOmega
)
{
    if constexpr(ugkwp::GasConservativeWallCapability<GasState>::value){
        if(ugkwp::conservativeWallCellReady(s,c)){
            sourceK=s.gasBoundaryLayerModel.conservative.sourceK[c];
            sourceOmega=s.gasBoundaryLayerModel.conservative.sourceOmega[c];
            return true;
        }
    }
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
    if(!gasProducerScalarValid(s,c,kProduction,30025))return false;
    sourceK = s.rho[c]*
    (
        kProduction
      - (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0&&ugkwp::gasBoundaryLayerSlotReady(s,slot)){
            const auto& closure=s.gasBoundaryLayer.sst[slot];
            if(closure.kFluxCount>0){
                // The BVP combines its normal gradient with this stage's
                // resolved tangential gradient before limiting production.
                // Its projected source already contains full compression.
                sourceK=closure.nonDestructiveKSource
                    -s.rho[c]*s.k[c]*closure.kDestructionRate;
            }
        }
    }
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
    return true;
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
    if constexpr(ugkwp::GasConservativeWallCapability<GasState>::value){
        if(ugkwp::conservativeWallCellReady(s,c))return s.gasBoundaryLayerModel.conservative.sourceActivity[c];
    }
    if(ugkwp::gasBoundaryLayerOwnerSlot(s,c)<0)return fabs(netSource);
    const GPU_OPERATOR_REAL compression=GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0)
        *s.rho[c]*s.k[c]*divU;
    const GPU_OPERATOR_REAL destruction=s.sstCoefficients.betaStar*s.rho[c]
        *s.k[c]*s.omega[c];
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value){
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0&&ugkwp::gasBoundaryLayerSlotReady(s,slot)){
            const auto& closure=s.gasBoundaryLayer.sst[slot];
            if(closure.kFluxCount>0){
                // The projected BVP term already contains its own normal
                // compression. Bound the actual constant/profile term, the
                // resolved correction, and local linear destruction separately.
                // This is >= |netSource| and cannot turn negative by cancellation.
                const GPU_OPERATOR_REAL projectedDestruction=s.rho[c]*s.k[c]*closure.kDestructionRate;
                const GPU_OPERATOR_REAL resolved=netSource+projectedDestruction-closure.nonDestructiveKSource;
                return fmax(fabs(netSource),fabs(closure.nonDestructiveKSource)+fabs(resolved)+projectedDestruction);
            }
        }
    }
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
    const GPU_OPERATOR_TIME dt,
    const bool nativeRefinement=false
)
{
    GasState& s = *sp;
    // A prior producer already diagnosed the fatal; retain its evidence until completion.
    if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL oldVolume=s.V[c],newVolume=s.V[c];
    const bool moving=ugkwp::gasMovingGeometry(s);
    if(moving)
    {
        bool evaluated=false;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated){if(!recorded)asm("trap;");return;}
        }
    }
    if(nativeRefinement && moving && ugkwp::gasCellFailure(s,c)!=0)return;
    bool constrainedOmega = ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
    GPU_OPERATOR_REAL fluxK = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL fluxOmega = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL volumeFlux = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(!gasStencilRangeValid(s,c,start,count))return;
    if (count>0 && (start<0 || !s.cellFaceId)){if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if(!gasStencilFaceValid(s,c,f))return;



        if(!ugkwp::gasFinite(s.sstPhiRhoK[f]) || !ugkwp::gasFinite(s.sstPhiRhoOmega[f]))
        {
            gasCellPhysicalDiagnostic(s,c,27,ugkp::diagnostics::Code::NonFinite,
                !ugkwp::gasFinite(s.sstPhiRhoK[f])?s.sstPhiRhoK[f]:s.sstPhiRhoOmega[f],
                !ugkwp::gasFinite(s.sstPhiRhoK[f])?0:1);
            ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;
        }
        const int own=s.faceOwner[f],nei=coupledFaceNeighbour(s,f);
        if (own<0 || own>=s.nCells || nei>=s.nCells)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? -GPU_OPERATOR_R(1.0) : GPU_OPERATOR_R(1.0);
        fluxK += sign*s.sstPhiRhoK[f];
        fluxOmega += sign*s.sstPhiRhoOmega[f];
        // Convert finalized relative ALE mass flux back to absolute fluid
        // volume flux for compression; mesh motion is not a physical source.
        GPU_OPERATOR_REAL absoluteVolumeFlux=s.gasPhiRho[f]/sstFaceDensity(s, f);
        if(moving)
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
            bool evaluated=false;
            if(!ugkwp::gasGeometryFaceFrame(s,f,dt,frame,&evaluated))
            {
                const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
                if(nativeRefinement || !evaluated){if(!recorded)asm("trap;");return;}
            }
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
    if(!sstSourcesForCell(s, c, divU, sourceK, sourceOmega))return;
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))
            {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            // The projected normal BVP row and resolved tangential row have
            // already been assembled by sstSourcesForCell.
        }
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + sourceK);
    const GPU_OPERATOR_REAL deltaRhoOmega = dt*(fluxOmega*invV + sourceOmega);
    if(!ugkwp::gasFinite(deltaRhoK) || (!constrainedOmega && !ugkwp::gasFinite(deltaRhoOmega)))
    {
        gasCellPhysicalDiagnostic(s,c,27,ugkp::diagnostics::Code::NonFinite,
            !ugkwp::gasFinite(deltaRhoK)?deltaRhoK:deltaRhoOmega,!ugkwp::gasFinite(deltaRhoK)?0:1);
        ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;
    }
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
    if(!ugkwp::gasFinite(s.sstSourceNumber[c]))
    {
        using namespace ugkp::diagnostics;
        // This post-update activity ratio is an audit only. The next Courant
        // query recomputes its own stability predictor before reducing it.
        recordWarning(s.diagnostics,makeRecord(Code::AccuracyResidual,2611,27,0,
            EntityKind::Cell,c,-1,s.sstSourceNumber[c],s.sstMaxSourceNumber,
            s.sstSourceNumber[c],Quantity::Accuracy));
    }
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
        if(ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0 && s.sstSourceNumber[c]>s.sstMaxSourceNumber)
        {
            using namespace ugkp::diagnostics;
            recordWarning(s.diagnostics,makeRecord(Code::AccuracyResidual,2612,27,0,
                EntityKind::Cell,c,-1,s.sstSourceNumber[c],s.sstMaxSourceNumber,
                s.sstSourceNumber[c],Quantity::Accuracy));
        }
    // Explicit sources are evaluated from the old density and old inventory.
    // Faces already carry the common ALE mass flux, so add no mesh term here.
    if(moving)
    {
        const GPU_OPERATOR_REAL nextK=(s.rhoK[c]*oldVolume+dt*(fluxK+oldVolume*sourceK))/newVolume;
        const GPU_OPERATOR_REAL nextOmega=(s.rhoOmega[c]*oldVolume
            +(constrainedOmega?GPU_OPERATOR_R(0.0):dt*(fluxOmega+oldVolume*sourceOmega)))/newVolume;
        if(!ugkwp::gasFinite(nextK)||!ugkwp::gasFinite(nextOmega))
        {
            using namespace ugkp::diagnostics;
            const int component=!ugkwp::gasFinite(nextK)?0:1;
            if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2613,27,0,
                EntityKind::Cell,c,component,component==0?nextK:nextOmega));
            ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
            return;
        }
        s.rhoK[c]=clampMin(nextK,rhoKFloor*oldVolume/newVolume);
        s.rhoOmega[c]=clampMin(nextOmega,rhoOmegaFloor*oldVolume/newVolume);
        return;
    }
    if(!gasProducerScalarValid(s,c,s.rhoK[c]+deltaRhoK,30024)
        ||(!constrainedOmega && !gasProducerScalarValid(s,c,s.rhoOmega[c]+deltaRhoOmega,30100)))return;
    s.rhoK[c] =
        clampMin(s.rhoK[c] + deltaRhoK, rhoKFloor);
    // Both wall treatments constrain the adjacent-cell omega equation.
    // Suppress its RHS, then primitive recovery projects to the refreshed
    // target after k and gas density evolve. Other equations remain active.
    if (!constrainedOmega)
    {
        s.rhoOmega[c] = clampMin
        (
            s.rhoOmega[c] + deltaRhoOmega,
            rhoOmegaFloor
        );
    }
}
template<class GasState>
__global__ void computeGasCourantFieldKernel(GasState* sp, const GPU_OPERATOR_TIME dt, const bool nativeRefinement=false)
{
    GasState& s = *sp;
    (void)dt;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> meshFrame;
    if(ugkwp::gasMovingGeometry(s))
    {
        bool evaluated=false;
        if(!ugkwp::gasGeometryFaceFrame(s,f,dt,meshFrame,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordFaceFailure(s,f,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated)
            {
                s.gasPhiRhoE[f]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();
                if(s.sstConfigured!=0)s.gasPhiRho[f]=GPU_OPERATOR_R(0.0);
                if(!recorded)asm("trap;");return;
            }
        }
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
            s, f, mass, mx, my, mz, energy,nativeRefinement
        );
        s.gasPhiRho[f] = mass;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        gasStencilIndexFailure(s,own,own,30020);
        s.gasPhiRhoE[f] = GPU_OPERATOR_R(0.0);
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
            gasStencilIndexFailure(s,own,nei,30021);
            s.gasPhiRhoE[f] = GPU_OPERATOR_R(0.0);
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
        right = riemannBoundaryState(s, f, left,nullptr,nativeRefinement);
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
    if(!gasProducerScalarValid(s,own,unLeft,30022)||!gasProducerScalarValid(s,own,unRight,30101)
        ||!gasProducerScalarValid(s,own,aLeft,30102)||!gasProducerScalarValid(s,own,aRight,30103))
    {s.gasPhiRhoE[f]=GPU_OPERATOR_R(0.0);return;}
    const GPU_OPERATOR_REAL spectralRadius = fmax
    (
        fabs(unLeft) + aLeft,
        fabs(unRight) + aRight
    );
    const GPU_OPERATOR_REAL amaxSf = spectralRadius*area;

    s.gasPhiRhoE[f] = nativeRefinement && !ugkwp::gasFinite(amaxSf) ? OfGreat : amaxSf;
    if(!ugkwp::gasFinite(amaxSf))ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
}

template<class GasState>
__global__ void computeGasConvectiveCourantByCellKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt,
    const bool nativeRefinement=false
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
        bool evaluated=false;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated)
            {s.gasFluxPositivityScale[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();if(!recorded)asm("trap;");return;}
        }
        stabilityVolume=fmin(oldVolume,newVolume);
    }
    GPU_OPERATOR_REAL sumAmaxSf = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(!gasStencilRangeValid(s,c,start,count))return;
    if (count>0 && (start<0 || !s.cellFaceId)){if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if(!gasStencilFaceIndexValid(s,c,f))return;
        if(ugkwp::gasFaceFailure(s,f)==int(ugkwp::GasTransportCode::InvalidStorage))
            ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);
        sumAmaxSf += nativeRefinement?s.gasPhiRhoE[f]:s.gasPhiRhoE[f];
    }



    const GPU_OPERATOR_REAL co =
        GPU_OPERATOR_R(0.5)*dt*sumAmaxSf/clampMin(stabilityVolume, OfSmall);
    s.gasFluxPositivityScale[c] = nativeRefinement && !ugkwp::gasFinite(co) ? OfGreat : co;
    if(!ugkwp::gasFinite(co))ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
}

template<class GasState>
__global__ void computeGasDiffusionNumberKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_REAL targetMaxCo,
    const bool nativeRefinement=false
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
        bool evaluated=false;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated)
            {s.gasDiffusionNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();if(!recorded)asm("trap;");return;}
        }
        stabilityVolume=fmin(oldVolume,newVolume);
    }
    GPU_OPERATOR_REAL sum = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL maxMolecularDiffusivity=GPU_OPERATOR_R(0.0);
    bool molecularDiffusivityComputed=false;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(!gasStencilRangeValid(s,c,start,count))return;
    if (count>0 && (start<0 || !s.cellFaceId)){if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if(!gasStencilFaceIndexValid(s,c,f))return;
        if(f>=s.nInternalFaces && s.riemannBoundaryKind[f]==4)continue;
        if(!gasStencilFaceCouplingValid(s,c,f))return;
        const int boundaryKind = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? 0 : s.riemannBoundaryKind[f];
        if(boundaryKind==3 || (boundaryKind==1&&s.riemannBoundaryTFix[f]==0))continue;
        if((f<s.nInternalFaces || isPeriodicFace(s,f)) && !gasStencilFaceValid(s,c,f))return;
        const int other = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? (s.faceOwner[f] == c ? s.faceNeighbour[f] : s.faceOwner[f])
          : -1;
        if (other>=s.nCells)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
        const GPU_OPERATOR_REAL rhoFace = other >= 0
          ? GPU_OPERATOR_R(0.5)*(s.rho[c] + s.rho[other]) : s.rho[c];
        GPU_OPERATOR_REAL muTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL kTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL directWallHeatFlux = GPU_OPERATOR_R(0.0);
        int directWallHeatFluxActive = 0;
        // Match the ordinary face-flux channels. A selected wall returns its
        // independent physical exchange before those ordinary boundary flags;
        // retain its existing coefficient bounds even for a cold profile.
        const bool wallExchange=ugkwp::gasBoundaryLayerFaceSlot(s,f)>=0;

        gasFaceSubgridTransportProperties
        (
            s, f, c, other, boundaryKind, rhoFace,
            muTurbulent, kTurbulent,
            directWallHeatFlux, directWallHeatFluxActive,nativeRefinement
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
                    if(!molecularDiffusivityComputed)
                    {
                        if(s.gasSpecies.diffusivity)
                            for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                            {
                                if(!gasProducerScalarValid(s,c,s.gasSpecies.diffusivity[k],30026))return;
                                maxMolecularDiffusivity=fmax(maxMolecularDiffusivity,GPU_OPERATOR_R(2.0)*s.gasSpecies.diffusivity[k]);
                            }
                        molecularDiffusivityComputed=true;
                    }
                    speciesDiffusivity=maxMolecularDiffusivity;
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
    s.gasDiffusionNumber[c] = nativeRefinement && !ugkwp::gasFinite(equivalentCo) ? OfGreat : equivalentCo;
    if(!ugkwp::gasFinite(equivalentCo))ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
}

// The selected wall supplies a prescribed normal omega diffusion flux on
// shared faces. This is an explicit forcing, not a coefficient multiplying the
// neighbouring macro omega. Reproduce only that forcing contribution here;
// the existing diffusion bound continues to cover the ordinary coefficients.
template<class GasState>
__device__ bool sstPublishedOmegaForcingForFace
(
    const GasState& s,const int f,const int own,const int nei,
    GPU_OPERATOR_REAL& forcing
)
{
    forcing=GPU_OPERATOR_R(0);
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        if(nei<0||!ugkwp::gasBoundaryLayerEnabled(s))return true;
        if constexpr(ugkwp::GasConservativeWallCapability<GasState>::value)
        {
            const auto& d=s.gasBoundaryLayerModel.conservative;
            if(d.enabled&&d.faceSlot&&d.faceSlot[f]>=0&&d.active[d.faceSlot[f]]
                &&ugkwp::gasBoundaryLayerSlotReady(s,d.faceSlot[f]))return true;
        }
        if(ugkwp::gasBoundaryLayerOwnerSlot(s,own)<0
            &&ugkwp::gasBoundaryLayerOwnerSlot(s,nei)<0)return true;
        const GPU_OPERATOR_REAL area=s.magSf[f];
        const GPU_OPERATOR_REAL unit[3]={s.Sfx[f]/area,s.Sfy[f]/area,s.Sfz[f]/area};
        GPU_OPERATOR_REAL mappedX,mappedY,mappedZ;
        periodicMappedCellCentre(s,f,nei,mappedX,mappedY,mappedZ);
        int contributors=0;
        for(int side=0;side<2;++side)
        {
            const int cell=side==0?own:nei;
            const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,cell);
            if(slot<0)continue;
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))return false;
            const auto& closure=s.gasBoundaryLayer.sst[slot];
            if(closure.kFluxCount<=0)continue;
            const auto* n=closure.normal;
            const GPU_OPERATOR_REAL alignment=unit[0]*n[0]+unit[1]*n[1]+unit[2]*n[2];
            const GPU_OPERATOR_REAL cx=side==0?s.Cx[own]:mappedX,
                cy=side==0?s.Cy[own]:mappedY,cz=side==0?s.Cz[own]:mappedZ;
            const GPU_OPERATOR_REAL y=closure.ownerDistance
                +(s.faceCx[f]-cx)*n[0]+(s.faceCy[f]-cy)*n[1]+(s.faceCz[f]-cz)*n[2];
            forcing+=ugkwp::boundaryLayerSstFlux(closure,y,true)*alignment*area;
            ++contributors;
        }
        if(contributors)forcing/=GPU_OPERATOR_REAL(contributors);
    }
    return true;
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
    const bool requireWallProfile=true,
    const bool nativeRefinement=false
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
        bool evaluated=false;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated)
            {s.sstSourceNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();if(!recorded)asm("trap;");return;}
        }
        stabilityVolume=fmin(oldVolume,newVolume);sourceVolume=oldVolume;
    }
    GPU_OPERATOR_REAL diffusionRate = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL publishedOmegaForcingActivity = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL minimumVolumeFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL maximumVolumeFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL meshVolumeFlux = GPU_OPERATOR_R(0.0);
    bool constrainedOmega = ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(!gasStencilRangeValid(s,c,start,count))return;
    if (count>0 && (start<0 || !s.cellFaceId)){if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if(!gasStencilFaceIndexValid(s,c,f))return;
        if(f>=s.nInternalFaces
            && (s.riemannBoundaryKind[f]==3 || s.riemannBoundaryKind[f]==4))continue;
        if(!gasStencilFaceValid(s,c,f))return;
        const int own=s.faceOwner[f],nei=coupledFaceNeighbour(s,f);
        if (own<0 || own>=s.nCells || nei>=s.nCells)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
        const GPU_OPERATOR_REAL outwardSign = s.faceOwner[f] == c
          ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        const GPU_OPERATOR_REAL outwardVolumeFlux =
            outwardSign*sstPredictorMassFlux(s, f)/sstFaceDensity(s, f);
        if(ugkwp::gasMovingGeometry(s))
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
            bool evaluated=false;
            if(!ugkwp::gasGeometryFaceFrame(s,f,dt,frame,&evaluated))
            {
                const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
                if(nativeRefinement || !evaluated)
                {s.sstSourceNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();if(!recorded)asm("trap;");return;}
            }
            meshVolumeFlux+=outwardSign*frame.meshVolumeRate;
        }
        minimumVolumeFlux += fmin(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        maximumVolumeFlux += fmax(outwardVolumeFlux, GPU_OPERATOR_R(0.0));
        constrainedOmega = constrainedOmega ||
        (
            (s.sstWallTreatment == 0 || s.sstWallTreatment == 1 || s.sstWallTreatment == 2)
         && f >= s.nInternalFaces && s.riemannBoundaryKind[f] == 2
        );
        if(requireWallProfile&&!constrainedOmega&&nei>=0)
        {
            GPU_OPERATOR_REAL publishedForcing;
            if(!sstPublishedOmegaForcingForFace(s,f,own,nei,publishedForcing))
            {
                s.sstSourceNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();
                if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
                return;
            }
            if(!ugkwp::gasFinite(publishedForcing))
            {
                s.sstSourceNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();
                ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
                return;
            }
            publishedOmegaForcingActivity+=fabs(publishedForcing);
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
                  ? s.riemannBoundaryUx[f] : GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
                  ? s.riemannBoundaryUy[f] : GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
                  ? s.riemannBoundaryUz[f] : GPU_OPERATOR_R(0.0);
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
                    s.sstWallE, s.diagnostics, f
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
    if(!sstSourcesForCell
    (
        s, c, GPU_OPERATOR_R(0.0), sourceKAtZeroDiv, sourceOmegaAtZeroDiv
    ))return;
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
            {s.sstSourceNumber[c]=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
            if(ugkwp::gasBoundaryLayerSlotReady(s,slot)
                && s.gasBoundaryLayer.sst[slot].kFluxCount>0)
            {
                // The full projected source already includes compression and
                // is independent of the macro face-divergence estimate.
                sourceKBound=sstKSourceActivity(s,c,sourceKAtZeroDiv,GPU_OPERATOR_R(0.0));
            }
            else
            {
                // Cold outer estimates and nonvolume closures still use the
                // coarse SST source with its resolved compression term.
                sourceKBound=fmax(
                    sstKSourceActivity(s,c,sourceKAtZeroDiv-compressionK*minimumDivU,minimumDivU),
                    sstKSourceActivity(s,c,sourceKAtZeroDiv-compressionK*maximumDivU,maximumDivU));
            }
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
        constrainedOmega ? GPU_OPERATOR_R(0.0) : fabs(dt)
          *(sourceOmegaBound+publishedOmegaForcingActivity*invVolume)
       /clampMin(s.rhoOmega[c], s.rho[c]*s.sstOmegaMin)
    );
    const GPU_OPERATOR_REAL equivalentCo = targetMaxCo*fmax
    (
        diffusionNumber/clampMin(s.maxDiffusionNumber, OfSmall),
        sourceNumber/clampMin(s.sstMaxSourceNumber, OfSmall)
    );
    s.sstSourceNumber[c] = nativeRefinement && !ugkwp::gasFinite(equivalentCo) ? OfGreat : equivalentCo;
    if(!ugkwp::gasFinite(equivalentCo))ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
}

template<class GasState>
__global__ void applyGasFluxDivergenceByCellKernel(GasState* sp, const GPU_OPERATOR_TIME dt, const bool nativeRefinement=false)
{
    GasState& s = *sp;
    // A prior producer already diagnosed the fatal; retain its evidence until completion.
    if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
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
    bool repairableFlux=true;

    constexpr GPU_OPERATOR_REAL roundoff=GPU_OPERATOR_REAL(0x1p-52);
    if constexpr(Ns>0)
        if(ugkwp::mixtureGasActive(s))
        {
            if(!s.gasSpecies.cellStatus){asm("trap;");return;}
            if(nativeRefinement && s.gasSpecies.cellStatus[c]!=0)return;
        }

    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(!gasStencilRangeValid(s,c,start,count))return;
    if (count>0 && (start<0 || !s.cellFaceId)){if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
    for (int i = 0; i < count; ++i)
    {
        const int faceI = s.cellFaceId[start + i];
        if(!gasStencilFaceValid(s,c,faceI))return;

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


            if(!s.gasSpecies.faceStatus)
            {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return;}
            if(s.gasSpecies.faceStatus[faceI]!=0)
            {ugkwp::gasRecordCellFailure(s,c,static_cast<ugkwp::GasTransportCode>(s.gasSpecies.faceStatus[faceI]));if(nativeRefinement)return;}
            if(!s.gasSpecies.flux)
            {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return;}
            GPU_OPERATOR_REAL faceSum=GPU_OPERATOR_R(0.0),faceMagnitude=GPU_OPERATOR_R(0.0);
            for(int k=0;k<Ns;++k)
            {
                const GPU_OPERATOR_REAL flux=s.gasSpecies.flux[k*s.nFaces+faceI];
                if(!ugkwp::gasFinite(flux))
                {gasCellPhysicalDiagnostic(s,c,28,ugkp::diagnostics::Code::NonFinite,flux,k);ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
                trial[k]+=sign*flux;faceSum+=flux;faceMagnitude+=fabs(flux);
            }
            const GPU_OPERATOR_REAL massFlux=s.gasPhiRho[faceI];
            const GPU_OPERATOR_REAL faceBound=GPU_OPERATOR_R(8*Ns)*roundoff
                *fmax(fabs(massFlux),faceMagnitude);
            if(!ugkwp::gasFinite(massFlux))
            {gasCellPhysicalDiagnostic(s,c,28,ugkp::diagnostics::Code::NonFinite,massFlux);
                ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
            if(!ugkwp::gasFinite(faceSum)||!ugkwp::gasFinite(faceMagnitude))
                gasCellResidualDiagnostic(s,c,28,!ugkwp::gasFinite(faceSum)?faceSum:faceMagnitude,massFlux);
            else if(fabs(faceSum-massFlux)>faceBound)gasCellResidualDiagnostic(s,c,28,faceSum,massFlux);
            repairableFlux=repairableFlux&&ugkwp::gasFinite(faceSum)&&ugkwp::gasFinite(faceMagnitude)
                &&fabs(faceSum-massFlux)<=faceBound;
            speciesFluxMagnitude+=faceMagnitude;
        }
        dRho += sign*s.gasPhiRho[faceI];
        dRhoUx += sign*s.gasPhiRhoUx[faceI];
        dRhoUy += sign*s.gasPhiRhoUy[faceI];
        dRhoUz += sign*s.gasPhiRhoUz[faceI];
        dRhoE += Energy(sign)*Energy(s.gasPhiRhoE[faceI]);
    }

    GPU_OPERATOR_REAL scale;
    Energy energyAfter;
    GPU_OPERATOR_REAL densityRatio=GPU_OPERATOR_R(1.0);
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL oldVolume,newVolume;
        bool evaluated=false;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,oldVolume,newVolume,&evaluated))
        {
            const bool recorded=ugkwp::gasRecordCellFailure(s,c,evaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage);
            if(nativeRefinement || !evaluated){if(!recorded)asm("trap;");return;}
        }
        if(nativeRefinement && ugkwp::gasCellFailure(s,c)!=0)return;
        for(int i=0;i<count;++i)
        {
            const int f=s.cellFaceId[start+i];
            if(!gasStencilFaceIndexValid(s,c,f))return;

            if(ugkwp::gasFaceFailure(s,f)!=0)
            {if(!ugkwp::gasRecordCellFailure(s,c,static_cast<ugkwp::GasTransportCode>(ugkwp::gasFaceFailure(s,f))) && nativeRefinement)asm("trap;");if(nativeRefinement)return;}
        }
        densityRatio=oldVolume/newVolume;scale=dt/newVolume;
        energyAfter=s.rhoE[c]*densityRatio+scale*dRhoE;

    }
    else
    {
        const bool mixture=ugkwp::mixtureGasActive(s);
        if constexpr(Ns>0)
            if(mixture && (!ugkwp::gasFinite(s.V[c]) || s.V[c]==GPU_OPERATOR_R(0.0)))
            {gasCellPhysicalDiagnostic(s,c,28,!ugkwp::gasFinite(s.V[c])?ugkp::diagnostics::Code::NonFinite:ugkp::diagnostics::Code::ZeroDenominator,s.V[c],-1,ugkp::diagnostics::Quantity::Denominator);return;}
        const GPU_OPERATOR_REAL volume=mixture?s.V[c]:clampMin(s.V[c],s.rhoMin);
        scale=dt/volume;
        Energy energyScale;
        energyScale=scale;
        energyAfter=s.rhoE[c]+energyScale*dRhoE;
    }
    const GPU_OPERATOR_REAL rhoBefore = s.rho[c];
    const GPU_OPERATOR_REAL rhoAfter = rhoBefore*densityRatio + scale*dRho;
    const GPU_OPERATOR_REAL momentumXAfter=s.rhoUx[c]*densityRatio+scale*dRhoUx;
    const GPU_OPERATOR_REAL momentumYAfter=s.rhoUy[c]*densityRatio+scale*dRhoUy;
    const GPU_OPERATOR_REAL momentumZAfter=s.rhoUz[c]*densityRatio+scale*dRhoUz;
    if(!ugkwp::gasFinite(rhoAfter)||!ugkwp::gasFinite(momentumXAfter)||!ugkwp::gasFinite(momentumYAfter)
        ||!ugkwp::gasFinite(momentumZAfter)||!ugkwp::gasEnergyFinite(energyAfter))
    {
        const double values[5]={double(rhoAfter),double(momentumXAfter),double(momentumYAfter),double(momentumZAfter),double(energyAfter)};
        for(int k=0;k<5;++k)if(!ugkwp::gasFinite(values[k])){gasCellPhysicalDiagnostic(s,c,28,ugkp::diagnostics::Code::NonFinite,values[k],k);break;}
        ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;
    }
    if(ugkwp::mixtureGasActive(s) && rhoAfter<=GPU_OPERATOR_R(0))
    {gasCellPhysicalDiagnostic(s,c,28,rhoAfter==GPU_OPERATOR_R(0)?ugkp::diagnostics::Code::ZeroDenominator:ugkp::diagnostics::Code::InvalidPhysicalState,rhoAfter,-1,ugkp::diagnostics::Quantity::Density);return;}
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            if(!s.gasSpecies.rho)
            {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return;}
            for(int k=0;k<Ns;++k)
            {
                trial[k]=s.gasSpecies.rho[k*s.nCells+c]*densityRatio+scale*trial[k];
                if(!ugkwp::gasFinite(trial[k]))
                {gasCellPhysicalDiagnostic(s,c,28,ugkp::diagnostics::Code::NonFinite,trial[k],k,ugkp::diagnostics::Quantity::Density);
                    ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
            }






            GPU_OPERATOR_REAL beforeSum=GPU_OPERATOR_R(0.0),afterSum=GPU_OPERATOR_R(0.0);
            int dependent=Ns-1;
            bool repairable=repairableFlux;
            for(int k=0;k<Ns;++k)
            {
                const GPU_OPERATOR_REAL previous=s.gasSpecies.rho[k*s.nCells+c];
                if(!ugkwp::gasFinite(previous)){gasCellPhysicalDiagnostic(s,c,28,ugkp::diagnostics::Code::NonFinite,previous,k,ugkp::diagnostics::Quantity::Density);return;}
                repairable=repairable&&ugkwp::gasFinite(previous)&&previous>=GPU_OPERATOR_R(0.0)
                    &&trial[k]>=GPU_OPERATOR_R(0.0);
                beforeSum+=previous;
                afterSum+=trial[k];
            }
            // Choose the largest carrier from the unmodified state once.
            for(int k=0;k<Ns;++k)if(trial[k]>trial[dependent])dependent=k;


            const GPU_OPERATOR_REAL bound=(GPU_OPERATOR_R(8*Ns)+GPU_OPERATOR_R(2)*count)*roundoff
                *(fmax(rhoAfter,rhoBefore*densityRatio)+fabs(scale)*speciesFluxMagnitude);
            repairable=repairable&&ugkwp::gasFinite(rhoBefore)&&rhoBefore>GPU_OPERATOR_R(0.0)
                &&rhoAfter>GPU_OPERATOR_R(0.0)&&ugkwp::gasFinite(beforeSum)&&ugkwp::gasFinite(afterSum)&&ugkwp::gasFinite(bound)
                &&ugkwp::gasFinite(s.gasSpecies.densityClosureTolerance)&&s.gasSpecies.densityClosureTolerance>=GPU_OPERATOR_R(0.0)
                &&fabs(beforeSum-rhoBefore)<=s.gasSpecies.densityClosureTolerance*rhoBefore;



            if(!ugkwp::gasFinite(beforeSum)||!ugkwp::gasFinite(afterSum))
            {gasCellResidualDiagnostic(s,c,28,!ugkwp::gasFinite(beforeSum)?beforeSum:afterSum,rhoAfter);repairable=false;}
            if(fabs(afterSum-rhoAfter)>s.gasSpecies.densityClosureTolerance*fabs(rhoAfter))
                gasCellResidualDiagnostic(s,c,28,afterSum,rhoAfter);
            const GPU_OPERATOR_REAL targetSum=rhoAfter+(beforeSum-rhoBefore)*densityRatio;
            if(!ugkwp::gasFinite(targetSum)){gasCellResidualDiagnostic(s,c,28,targetSum,rhoAfter);repairable=false;}
            if(afterSum!=targetSum&&(!repairable||fabs(afterSum-targetSum)>bound))
                gasCellResidualDiagnostic(s,c,28,afterSum,targetSum);
            if(repairable&&afterSum!=targetSum&&fabs(afterSum-targetSum)<=bound)
            {
                GPU_OPERATOR_REAL independent=GPU_OPERATOR_R(0.0);
                for(int k=0;k<Ns;++k)if(k!=dependent)independent+=trial[k];
                GPU_OPERATOR_REAL replacement=targetSum-independent;


                GPU_OPERATOR_REAL publishedSum=GPU_OPERATOR_R(0.0);
                for(int k=0;k<Ns;++k)publishedSum+=k==dependent?replacement:trial[k];
                if(ugkwp::gasFinite(replacement)&&replacement>=GPU_OPERATOR_R(0.0)
                    &&fabs(replacement-trial[dependent])<=bound&&ugkwp::gasFinite(publishedSum)
                    &&fabs(publishedSum-rhoAfter)<=s.gasSpecies.densityClosureTolerance*rhoAfter)
                    trial[dependent]=replacement;
                else gasCellResidualDiagnostic(s,c,28,afterSum,targetSum);
            }

            for(int k=0;k<Ns;++k)s.gasSpecies.rho[k*s.nCells+c]=trial[k];
        }
    }
    s.rho[c]=rhoAfter;
    s.rhoUx[c]=momentumXAfter;s.rhoUy[c]=momentumYAfter;s.rhoUz[c]=momentumZAfter;
    s.rhoE[c]=energyAfter;
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
    }
}

template<class GasState>
__global__ void blendGasConservativeStateKernel
(
    GasState* sp,
    const double energyInitialWeight,
    const double energyStageWeight,
    const bool nativeRefinement=false
)
{
    GasState& s = *sp;
    // A prior producer already diagnosed the fatal; retain its evidence until completion.
    if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
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
        if(!s.gasSpecies.cellStatus){asm("trap;");return;}
        if(nativeRefinement && s.gasSpecies.cellStatus[c]!=0)return;
        if(!s.gasSpecies.rho || !s.gasSpecies.initial)
        {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return;}

        GPU_OPERATOR_REAL sumInitial=GPU_OPERATOR_R(0),sumStage=GPU_OPERATOR_R(0),sumBlend=GPU_OPERATOR_R(0);
        int dependent=reviewNs-1;
        bool repairable=true;
        for(int k=0;k<reviewNs;++k)
        {
            const int i=k*s.nCells+c;
            const GPU_OPERATOR_REAL a=s.gasSpecies.initial[i],b=s.gasSpecies.rho[i];
            if(!ugkwp::gasFinite(a)||!ugkwp::gasFinite(b))
            {gasCellPhysicalDiagnostic(s,c,29,ugkp::diagnostics::Code::NonFinite,!ugkwp::gasFinite(a)?a:b,k,ugkp::diagnostics::Quantity::Density);return;}
            repairable=repairable&&a>=GPU_OPERATOR_R(0)&&b>=GPU_OPERATOR_R(0);
            sumInitial+=a;sumStage+=b;blendedSpecies[k]=initialWeight*a+stageWeight*b;sumBlend+=blendedSpecies[k];
            if(!ugkwp::gasFinite(blendedSpecies[k]))
            {gasCellPhysicalDiagnostic(s,c,29,ugkp::diagnostics::Code::NonFinite,blendedSpecies[k],k,ugkp::diagnostics::Quantity::Density);return;}
        }
        if(!ugkwp::gasFinite(sumInitial)||!ugkwp::gasFinite(sumStage)||!ugkwp::gasFinite(sumBlend))
        {gasCellResidualDiagnostic(s,c,29,!ugkwp::gasFinite(sumInitial)?sumInitial:!ugkwp::gasFinite(sumStage)?sumStage:sumBlend,blendedDensity);repairable=false;}
        const GPU_OPERATOR_REAL tol=s.gasSpecies.densityClosureTolerance;
        repairable=repairable&&ugkwp::gasFinite(tol)&&tol>=GPU_OPERATOR_R(0)&&ugkwp::gasFinite(blendedDensity)&&blendedDensity>GPU_OPERATOR_R(0)
            &&ugkwp::gasFinite(s.rhoNext[c])&&s.rhoNext[c]>GPU_OPERATOR_R(0)&&ugkwp::gasFinite(s.rho[c])&&s.rho[c]>GPU_OPERATOR_R(0)
            &&fabs(sumInitial-s.rhoNext[c])<=tol*s.rhoNext[c]&&fabs(sumStage-s.rho[c])<=tol*s.rho[c];


        if(fabs(sumBlend-blendedDensity)>tol*fabs(blendedDensity))gasCellResidualDiagnostic(s,c,29,sumBlend,blendedDensity);
        const GPU_OPERATOR_REAL target=blendedDensity+initialWeight*(sumInitial-s.rhoNext[c])+stageWeight*(sumStage-s.rho[c]);
        constexpr GPU_OPERATOR_REAL eps=GPU_OPERATOR_REAL(0x1p-52);
        const GPU_OPERATOR_REAL bound=GPU_OPERATOR_R(8*reviewNs)*eps*(fabs(initialWeight*s.rhoNext[c])+fabs(stageWeight*s.rho[c]));
        if(!ugkwp::gasFinite(target)){gasCellResidualDiagnostic(s,c,29,target,blendedDensity);repairable=false;}
        if(sumBlend!=target&&(!repairable||fabs(sumBlend-target)>bound))gasCellResidualDiagnostic(s,c,29,sumBlend,target);
        if(repairable&&fabs(sumBlend-target)<=bound&&sumBlend!=target)
        {
            // This carrier is fixed before evaluating the closure correction.
            for(int k=0;k<reviewNs;++k)if(blendedSpecies[k]>blendedSpecies[dependent])dependent=k;
            GPU_OPERATOR_REAL independent=GPU_OPERATOR_R(0);for(int k=0;k<reviewNs;++k)if(k!=dependent)independent+=blendedSpecies[k];
            GPU_OPERATOR_REAL replacement=target-independent;
            GPU_OPERATOR_REAL published=GPU_OPERATOR_R(0);
            for(int k=0;k<reviewNs;++k)published+=k==dependent?replacement:blendedSpecies[k];
            if(ugkwp::gasFinite(replacement)&&replacement>=GPU_OPERATOR_R(0)&&fabs(replacement-blendedSpecies[dependent])<=bound
                &&ugkwp::gasFinite(published)&&fabs(published-blendedDensity)<=tol*blendedDensity)
                blendedSpecies[dependent]=replacement;
            else gasCellResidualDiagnostic(s,c,29,sumBlend,target);
        }

    }
    const GPU_OPERATOR_REAL blendedMomentumX=initialWeight*s.rhoUxNext[c]+stageWeight*s.rhoUx[c];
    const GPU_OPERATOR_REAL blendedMomentumY=initialWeight*s.rhoUyNext[c]+stageWeight*s.rhoUy[c];
    const GPU_OPERATOR_REAL blendedMomentumZ=initialWeight*s.rhoUzNext[c]+stageWeight*s.rhoUz[c];
    Energy blendedEnergy=s.rhoE[c];
    {
        if (!(s.rhoENext[c]==s.rhoE[c]
            && initialWeight+stageWeight==GPU_OPERATOR_R(1)))
            blendedEnergy=initialWeight*s.rhoENext[c]+stageWeight*s.rhoE[c];
    }
    const GPU_OPERATOR_REAL blendedK=s.sstConfigured!=0
        ?initialWeight*s.rhoKInitial[c]+stageWeight*s.rhoK[c]:GPU_OPERATOR_R(0);
    const GPU_OPERATOR_REAL blendedOmega=s.sstConfigured!=0
        ?initialWeight*s.rhoOmegaInitial[c]+stageWeight*s.rhoOmega[c]:GPU_OPERATOR_R(0);
    const double outputs[7]={double(blendedDensity),double(blendedMomentumX),double(blendedMomentumY),double(blendedMomentumZ),double(blendedEnergy),double(blendedK),double(blendedOmega)};
    for(int k=0;k<7;++k)if(!ugkwp::gasFinite(outputs[k]))
    {gasCellPhysicalDiagnostic(s,c,29,ugkp::diagnostics::Code::NonFinite,outputs[k],k);return;}
    s.rho[c]=blendedDensity;
    s.rhoUx[c]=blendedMomentumX;s.rhoUy[c]=blendedMomentumY;s.rhoUz[c]=blendedMomentumZ;
    s.rhoE[c]=blendedEnergy;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if (ugkwp::mixtureGasActive(s))
            for (int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
            {
                const int i=k*s.nCells+c;
                s.gasSpecies.rho[i]=blendedSpecies[k];
            }
    if (s.sstConfigured != 0)
    {
        s.rhoK[c]=blendedK;
        s.rhoOmega[c]=blendedOmega;
    }
}

template<class GasState>
__device__ void recoverGasPrimitiveCell(GasState& s, const int c, const bool nativeRefinement=false)
{
    // A prior producer already diagnosed the fatal; retain its evidence until completion.
    if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        auto& species = s.gasSpecies;
        if (species.mode != ugkwp::GasMode::SingleLegacy)
        {


            if (!species.cellStatus) { asm("trap;"); return; }
            if (nativeRefinement && species.cellStatus[c] != 0) return;
            if (!species.rho || !species.soundSpeed || !species.heatCapacity
                || !species.gasConstant)
            {
                ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);
                return;
            }
            using Energy = ugkwp::GasEnergyScalar<GasState>;
            constexpr int Ns = ugkwp::GasStateTraits<GasState>::speciesCount;
            Energy partialDensity[Ns];
            Energy densitySum = Energy(0);
            Energy gasConstantDensity = Energy(0.0);
            int failure=species.cellStatus[c];
            const Energy density = s.rho[c];
            for (int k = 0; k < Ns; ++k)
            {
                partialDensity[k] = species.rho[k*s.nCells+c];
                if (!ugkwp::gasFinite(partialDensity[k]))
                {gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::NonFinite,partialDensity[k],k,ugkp::diagnostics::Quantity::Density);return;}
                densitySum+=partialDensity[k];
            }
            if(!ugkwp::gasFinite(density) || density<=Energy(0))
            {gasCellPhysicalDiagnostic(s,c,30,!ugkwp::gasFinite(density)?ugkp::diagnostics::Code::NonFinite:
                density==Energy(0)?ugkp::diagnostics::Code::ZeroDenominator:ugkp::diagnostics::Code::InvalidPhysicalState,
                density,-1,ugkp::diagnostics::Quantity::Density);return;}
            for(int k=0;k<Ns;++k)
            {
                const Energy fraction=partialDensity[k]/density;
                if(!ugkwp::gasFinite(fraction))
                {gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::NonFinite,fraction,k,ugkp::diagnostics::Quantity::MassFraction);return;}
                const Energy clipped=fraction<Energy(0)?Energy(0):fraction>Energy(1)?Energy(1):fraction;
                if(clipped!=fraction){
                    using namespace ugkp::diagnostics;
                    recordWarning(s.diagnostics,makeRecord(Code::FiniteYClip,2622,30,0,EntityKind::Cell,c,k,double(fraction),double(clipped),double(clipped),Quantity::MassFraction));
                    partialDensity[k]=density*clipped;
                }
            }
            // This pre-clip total is conservation-audit data only. Thermodynamics
            // below consumes the individually validated, finite clipped densities.
            if(fabs(densitySum-density)>species.densityClosureTolerance*density)
                gasCellResidualDiagnostic(s,c,30,densitySum,density);
            const Energy ux = Energy(s.rhoUx[c])/density;
            const Energy uy = Energy(s.rhoUy[c])/density;
            const Energy uz = Energy(s.rhoUz[c])/density;
            const Energy internalEnergy = s.rhoE[c]
                - Energy(0.5)*density*(ux*ux+uy*uy+uz*uz);
            Energy temperature = s.Tgas[c];
            bool temperatureEvaluated=false;
            const auto status = nativeRefinement
                ? ugkwp::invertMixtureEnergy<true>(partialDensity,internalEnergy,species.thermo,
                    ugkwp::castThermoControls<Energy>(species.thermoControls),temperature,&temperatureEvaluated)
                : ugkwp::invertMixtureEnergy(partialDensity,internalEnergy,species.thermo,
                    ugkwp::castThermoControls<Energy>(species.thermoControls),temperature,&temperatureEvaluated);
            if (status != ugkwp::ThermoStatus::Success)
            {
                gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::InvalidPhysicalState,int(status),-1,ugkp::diagnostics::Quantity::Unspecified);
                ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidThermodynamics);return;
            }
            Energy cpDensity = Energy(0.0);
            for (int k = 0; k < Ns; ++k)
            {
                gasConstantDensity += partialDensity[k]*ugkwp::universalGasConstant<Energy>()
                    /species.thermo.species[k].molarMass;
                if(nativeRefinement)
                    cpDensity += partialDensity[k]*ugkwp::speciesCp(k, temperature, species.thermo);
                else
                {
                    ugkwp::gasThermoDetail::SpeciesValues<Energy> values;
                    bool propertyEvaluated=false;
                    const auto property=ugkwp::gasThermoDetail::evaluateSpecies(k,temperature,species.thermo,values,&propertyEvaluated);
                    if(!propertyEvaluated)
                    {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return;}
                    if(property!=ugkwp::ThermoStatus::Success)
                    {gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::InvalidPhysicalState,int(property),k);return;}
                    cpDensity += partialDensity[k]*values.cp;
                }
            }
            const Energy R = gasConstantDensity/density;
            const Energy cp = cpDensity/density;
            const Energy pressure = gasConstantDensity*temperature;
            const Energy a2 = cp/(cp-R)*R*temperature;
            if (!ugkwp::gasFinite(ux) || !ugkwp::gasFinite(uy) || !ugkwp::gasFinite(uz)
                || !ugkwp::gasFinite(pressure)
                || !ugkwp::gasFinite(sqrt(a2))
                || (nativeRefinement && (!(pressure>Energy(0)) || !ugkwp::gasFinite(a2) || !(a2>Energy(0)))))
            {
                gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::InvalidPhysicalState,a2);
                ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidThermodynamics);return;
            }

            s.Ux[c] = ux; s.Uy[c] = uy; s.Uz[c] = uz;
            s.p[c] = pressure; s.Tgas[c] = temperature;
            species.soundSpeed[c] = sqrt(a2);
            species.heatCapacity[c] = cp; species.gasConstant[c] = R;
            ugkwp::gasRecordCellFailure(s,c,static_cast<ugkwp::GasTransportCode>(failure));
            return;
        }
    }
    {
        // Preserve the established homogeneous legacy numerical path.
    const GPU_OPERATOR_REAL conserved[]={s.rho[c],s.rhoUx[c],s.rhoUy[c],s.rhoUz[c]};
    for(int k=0;k<4;++k)if(!gasProducerScalarValid(s,c,conserved[k],30010))return;
    if(!gasProducerScalarValid(s,c,s.rhoE[c],30124))return;
    if(ugkwp::gasMovingGeometry(s))
    {
        if(nativeRefinement && ugkwp::gasCellFailure(s,c)!=0)return;
        const GPU_OPERATOR_REAL density=s.rho[c];
        if(density==GPU_OPERATOR_R(0)){gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::ZeroDenominator,density);return;}
        const GPU_OPERATOR_REAL ux=s.rhoUx[c]/density,uy=s.rhoUy[c]/density,uz=s.rhoUz[c]/density;
        const GPU_OPERATOR_REAL internal=s.rhoE[c]-GPU_OPERATOR_R(0.5)*density*(ux*ux+uy*uy+uz*uz);
        const GPU_OPERATOR_REAL pressure=(s.gammaGas-GPU_OPERATOR_R(1.0))*internal;
        const GPU_OPERATOR_REAL temperature=pressure/(density*s.Rgas);
        if(!ugkwp::gasFinite(density)||!ugkwp::gasFinite(ux)||!ugkwp::gasFinite(uy)||!ugkwp::gasFinite(uz)
            ||!ugkwp::gasFinite(pressure)||!ugkwp::gasFinite(temperature))
        {
            gasCellPhysicalDiagnostic(s,c,30,ugkp::diagnostics::Code::NonFinite,
                !ugkwp::gasFinite(density)?density:!ugkwp::gasFinite(pressure)?pressure:temperature);
            ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;
        }
        if(!gasProducerScalarValid(s,c,temperature,30011,true))return;
        s.Ux[c]=ux;s.Uy[c]=uy;s.Uz[c]=uz;s.p[c]=pressure;s.Tgas[c]=temperature;
        return;
    }
    const GPU_OPERATOR_REAL rhoSafe =
        clampMin(s.rho[c], s.rhoMin);
    if(!gasProducerScalarValid(s,c,s.rhoUx[c]/rhoSafe,30012)
        ||!gasProducerScalarValid(s,c,s.rhoUy[c]/rhoSafe,30104)
        ||!gasProducerScalarValid(s,c,s.rhoUz[c]/rhoSafe,30105))return;
    const GPU_OPERATOR_REAL ux = s.rhoUx[c]/rhoSafe;
    const GPU_OPERATOR_REAL uy = s.rhoUy[c]/rhoSafe;
    const GPU_OPERATOR_REAL uz = s.rhoUz[c]/rhoSafe;
    const GPU_OPERATOR_REAL kinetic = ugkwp::gasLegacyKineticEnergy(rhoSafe, ux, uy, uz);
    if(!gasProducerScalarValid(s,c,ux,30106)||!gasProducerScalarValid(s,c,uy,30107)
        ||!gasProducerScalarValid(s,c,uz,30108)||!gasProducerScalarValid(s,c,kinetic,30109)
        ||!gasProducerScalarValid(s,c,s.rhoE[c]-kinetic,30110))return;
    // Preserve the established finite density and internal-energy floors.
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
        s.rhoE[c] - kinetic,
        minimumInternalEnergy
    );
    const GPU_OPERATOR_REAL p = (s.gammaGas - GPU_OPERATOR_R(1.0))*internalE;
    const GPU_OPERATOR_REAL T = p/(rhoSafe*s.Rgas);

    if(!gasProducerScalarValid(s,c,p,30013)||!gasProducerScalarValid(s,c,T,30111,true))return;
    const bool repairDensity=!ugkwp::gasFinite(s.rho[c]) || s.rho[c]<s.rhoMin;
    s.rho[c] = rhoSafe;
    {
        // Valid conserved momenta are inputs, not rounded primitive caches.
        if (repairDensity || !ugkwp::gasFinite(s.rhoUx[c]/rhoSafe)) s.rhoUx[c]=rhoSafe*ux;
        if (repairDensity || !ugkwp::gasFinite(s.rhoUy[c]/rhoSafe)) s.rhoUy[c]=rhoSafe*uy;
        if (repairDensity || !ugkwp::gasFinite(s.rhoUz[c]/rhoSafe)) s.rhoUz[c]=rhoSafe*uz;
    }
    // A valid double inventory is authoritative; subtract/add can lose one ULP.

    {
        if (!ugkwp::gasFinite(s.rhoE[c]-kinetic)
            || s.rhoE[c]-kinetic < minimumInternalEnergy)
            s.rhoE[c] = kinetic + internalE;
    }
    s.Ux[c] = ux;
    s.Uy[c] = uy;
    s.Uz[c] = uz;
    s.p[c] = p;
    s.Tgas[c] = T;
    }
}

template<class GasState>
__global__ void recoverGasPrimitivesKernel(GasState* sp, const bool nativeRefinement=false)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    recoverGasPrimitiveCell(s, c, nativeRefinement);
}
