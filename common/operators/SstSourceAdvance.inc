template<class GasState,class SourceReal>
__device__ bool sstSourceTermsForCell
(
    const GasState& s,
    const int c,
    const GPU_OPERATOR_REAL divU,
    SourceReal& sourceK,
    SourceReal& sourceOmega,
    GPU_OPERATOR_REAL& decayK,
    GPU_OPERATOR_REAL& decayOmega
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
    if (s.sstWallTreatment == 0)
    {
        const auto terms = ugkwp::sstLowReSources(
            s.rho[c], s.k[c], s.omega[c], divU, kProduction, gByNu, s2, s.sstF1[c], s.sstF2[c], cd,
            s.sstCoefficients);
        sourceK = terms.kProduction;
        sourceOmega = terms.omegaProduction;
        decayK = terms.kDecay;
        decayOmega = terms.omegaDecay;
        return true;
    }
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

template<class GasState,class SourceReal>
__device__ bool sstSourcesForCell(const GasState& s,int c,GPU_OPERATOR_REAL divU,SourceReal& sourceK,SourceReal& sourceOmega)
{
    GPU_OPERATOR_REAL decayK=0,decayOmega=0;
    if(!sstSourceTermsForCell(s,c,divU,sourceK,sourceOmega,decayK,decayOmega))return false;
    if(s.sstWallTreatment==0)
    {
        sourceK-=SourceReal(decayK)*s.rhoK[c];
        sourceOmega-=SourceReal(decayOmega)*s.rhoOmega[c];
    }
    return true;
}

template<class GasState>
__global__ void applySstFluxAndSourceKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME stepDt,
    const bool nativeRefinement=false,
    const bool implicitDestruction=true
)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    GasState& s = *sp;

    if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    const GPU_OPERATOR_TIME dt = ugkwp::gasCellTimeStep(s, c, stepDt);

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
    GPU_OPERATOR_REAL decayK = GPU_OPERATOR_R(0), decayOmega = GPU_OPERATOR_R(0);
    if(!sstSourceTermsForCell(s, c, divU, sourceK, sourceOmega, decayK, decayOmega))return;
    const GPU_OPERATOR_REAL physicalDt = GPU_OPERATOR_R(dt);
    const bool damped = s.sstWallTreatment == 0 && implicitDestruction;
    if (!damped)
    {
        sourceK -= decayK*s.rhoK[c];
        sourceOmega -= decayOmega*s.rhoOmega[c];
    }
    const GPU_OPERATOR_REAL diagonalK = GPU_OPERATOR_R(1) + physicalDt*decayK;
    const GPU_OPERATOR_REAL diagonalOmega = GPU_OPERATOR_R(1) + physicalDt*decayOmega;
    if(damped && (!gasProducerScalarValid(s,c,diagonalK,30024)
        ||(!constrainedOmega && !gasProducerScalarValid(s,c,diagonalOmega,30100))))return;
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);
        if(slot>=0)
        {
            if(!ugkwp::gasBoundaryLayerSlotReady(s,slot))
            {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}


        }
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    const GPU_OPERATOR_REAL deltaRhoK = physicalDt*(fluxK*invV + sourceK);
    const GPU_OPERATOR_REAL deltaRhoOmega = physicalDt*(fluxOmega*invV + sourceOmega);
    if(!ugkwp::gasFinite(deltaRhoK) || (!constrainedOmega && !ugkwp::gasFinite(deltaRhoOmega)))
    {
        gasCellPhysicalDiagnostic(s,c,27,ugkp::diagnostics::Code::NonFinite,
            !ugkwp::gasFinite(deltaRhoK)?deltaRhoK:deltaRhoOmega,!ugkwp::gasFinite(deltaRhoK)?0:1);
        ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;
    }
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;


    if(moving)
    {
        GPU_OPERATOR_REAL nextK=(s.rhoK[c]*oldVolume+physicalDt*(fluxK+oldVolume*sourceK))/newVolume;
        GPU_OPERATOR_REAL nextOmega=(s.rhoOmega[c]*oldVolume
            +(constrainedOmega?GPU_OPERATOR_R(0.0):physicalDt*(fluxOmega+oldVolume*sourceOmega)))
            /newVolume;
        if(damped){nextK/=diagonalK;if(!constrainedOmega)nextOmega/=diagonalOmega;}
        if(!ugkwp::gasFinite(nextK)||!ugkwp::gasFinite(nextOmega))
        {
            using namespace ugkp::diagnostics;
            const int component=!ugkwp::gasFinite(nextK)?0:1;
            if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2613,27,0,
                EntityKind::Cell,c,component,component==0?nextK:nextOmega));
            ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
            return;
        }
        gasInventoryFloorWarning(s,c,nextK,rhoKFloor*oldVolume/newVolume,0,2623,27);
        gasInventoryFloorWarning(s,c,nextOmega,rhoOmegaFloor*oldVolume/newVolume,1,2623,27);
        s.rhoK[c]=clampMin(nextK,rhoKFloor*oldVolume/newVolume);
        s.rhoOmega[c]=clampMin(nextOmega,rhoOmegaFloor*oldVolume/newVolume);
        return;
    }
    const GPU_OPERATOR_REAL nextK = damped ? (s.rhoK[c]+deltaRhoK)/diagonalK : s.rhoK[c]+deltaRhoK;
    if(!gasProducerScalarValid(s,c,nextK,30024))return;
    gasInventoryFloorWarning(s,c,nextK,rhoKFloor,0,2623,27);
    s.rhoK[c] = clampMin(nextK, rhoKFloor);
    if (!constrainedOmega)
    {
        GPU_OPERATOR_REAL nextOmega = damped ? (s.rhoOmega[c]+deltaRhoOmega)/diagonalOmega : s.rhoOmega[c]+deltaRhoOmega;
        if(!gasProducerScalarValid(s,c,nextOmega,30100))return;
        gasInventoryFloorWarning(s,c,nextOmega,rhoOmegaFloor,1,2623,27);
        s.rhoOmega[c] = clampMin(nextOmega, rhoOmegaFloor);
    }
}
