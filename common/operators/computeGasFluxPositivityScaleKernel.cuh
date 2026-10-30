#pragma once
#include "gasTransport/GasPseudoTime.H"
#include "gasTransport/PositiveInventory.H"
#include "GpuWallEnergySink.cuh"
#include "gasTransport/GasGeometryValidation.H"

// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasFluxPositivityScaleKernel(GasState* sp, const GPU_OPERATOR_TIME stepDt)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }



    const GPU_OPERATOR_TIME dt = ugkwp::gasCellTimeStep(s, c, stepDt);

    if(dt<=GPU_OPERATOR_R(0.0))
    {
        s.gasFluxPositivityScale[c]=GPU_OPERATOR_R(1.0);
        if constexpr(ugkwp::GasStateTraits<GasState>::speciesCount>0)
            if(ugkwp::mixtureGasActive(s))
                for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                    s.gasSpecies.positivityScale[k*s.nCells+c]=GPU_OPERATOR_R(1.0);
        return;
    }
    GPU_OPERATOR_REAL inventoryVolume;
    bool inventoryVolumeComputed=false;
    GPU_OPERATOR_REAL outgoingMassFlux = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    if(count>0 && (start<0 || !s.cellFaceId))
    {
        if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return;
    }
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const GPU_OPERATOR_REAL phi = finiteOr(s.gasPhiRho[f], GPU_OPERATOR_R(0.0));
        if (s.faceOwner[f] == c && phi > GPU_OPERATOR_R(0.0))
        {
            outgoingMassFlux += phi;
        }
        else if (s.faceNeighbour[f] == c && phi < GPU_OPERATOR_R(0.0))
        {
            outgoingMassFlux -= phi;
        }
    }

    GPU_OPERATOR_REAL scale = GPU_OPERATOR_R(1.0);
    if (outgoingMassFlux > GPU_OPERATOR_R(0.0) && dt > GPU_OPERATOR_R(0.0))
    {
        const GPU_OPERATOR_REAL* volumes;
        if constexpr(ugkwp::GasGeometryCapability<GasState>::value)
            volumes=ugkwp::gasMovingGeometry(s)?s.gasGeometry.oldVolume:s.V;
        else volumes=s.V;
        if(!volumes)
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
        inventoryVolume=volumes[c];
        if(!ugkwp::gasFinite(inventoryVolume))
        {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
        inventoryVolumeComputed=true;
        scale=ugkwp::inventoryOutflowScale(
            clampMin(s.rho[c]-s.rhoMin,GPU_OPERATOR_R(0)),inventoryVolume,
            GPU_OPERATOR_REAL(dt),outgoingMassFlux);
    }
    s.gasFluxPositivityScale[c] = scale;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
            {
                GPU_OPERATOR_REAL outflow=GPU_OPERATOR_R(0.0);
                for (int i=0; i<count; ++i)
                {
                    const int f=s.cellFaceId[start+i];
                    if (f<0 || f>=s.nFaces) continue;
                    const GPU_OPERATOR_REAL flux=s.gasSpecies.flux[k*s.nFaces+f];
                if(!ugkwp::gasFinite(flux))
                {
                    using namespace ugkp::diagnostics;
                    if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2673,32,0,EntityKind::Face,f,k,flux));
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);return;
                }
                    if (s.faceOwner[f]==c && flux>GPU_OPERATOR_R(0.0)) outflow+=flux;
                    else if (s.faceNeighbour[f]==c && flux<GPU_OPERATOR_R(0.0)) outflow-=flux;
                }
                if(!ugkwp::gasFinite(outflow))
                {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
                GPU_OPERATOR_REAL factor=GPU_OPERATOR_R(1.0);
                if (dt>GPU_OPERATOR_R(0.0) && outflow>GPU_OPERATOR_R(0.0))
                {
                    if(!inventoryVolumeComputed)
                    {
                        const GPU_OPERATOR_REAL* volumes;
                        if constexpr(ugkwp::GasGeometryCapability<GasState>::value)
                            volumes=ugkwp::gasMovingGeometry(s)?s.gasGeometry.oldVolume:s.V;
                        else volumes=s.V;
                        if(!volumes)
                        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return;}
                        inventoryVolume=volumes[c];
                        if(!ugkwp::gasFinite(inventoryVolume))
                        {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);return;}
                        inventoryVolumeComputed=true;
                    }
                    factor=ugkwp::inventoryOutflowScale(
                        s.gasSpecies.rho[k*s.nCells+c],inventoryVolume,
                        GPU_OPERATOR_REAL(dt),outflow);
                }
                s.gasSpecies.positivityScale[k*s.nCells+c]=factor;
            }
    }
}

template<class GasState>
__global__ void applyGasFluxPositivityScaleKernel(GasState* sp, const bool nativeRefinement=false)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const GPU_OPERATOR_REAL phi = finiteOr(s.gasPhiRho[f], GPU_OPERATOR_R(0.0));
    GPU_OPERATOR_REAL scale = GPU_OPERATOR_R(1.0);
    if (phi > GPU_OPERATOR_R(0.0))
    {
        const int own = s.faceOwner[f];
        if (own >= 0 && own < s.nCells)
        {
            scale = s.gasFluxPositivityScale[own];
        }
    }
    else if (phi < GPU_OPERATOR_R(0.0) && coupledFaceNeighbour(s, f) >= 0)
    {
        const int nei = s.faceNeighbour[f];
        if (nei >= 0 && nei < s.nCells)
        {
            scale = s.gasFluxPositivityScale[nei];
        }
    }

    scale = clampRange(finiteOr(scale, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
            {
                const GPU_OPERATOR_REAL flux=s.gasSpecies.flux[k*s.nFaces+f];
                if(!ugkwp::gasFinite(flux))
                {
                    using namespace ugkp::diagnostics;
                    if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2674,32,0,EntityKind::Face,f,k,flux));
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);return;
                }
                const int donor=flux>GPU_OPERATOR_R(0.0)?s.faceOwner[f]:coupledFaceNeighbour(s,f);
                if (flux!=GPU_OPERATOR_R(0.0) && donor>=0 && donor<s.nCells)
                    scale=fmin(scale,s.gasSpecies.positivityScale[k*s.nCells+donor]);
            }
            // A prescribed physical wall exchange cannot be silently reduced:
            // that would separate the gas update from its material/source input.
            // Net-zero counterflow still checks each gas-side species donor.
            if(ugkwp::gasBoundaryLayerFaceSlot(s,f)>=0)
            {
                if(scale<GPU_OPERATOR_R(1.0))
                {
                    using namespace ugkp::diagnostics;
                    recordWarning(s.diagnostics,makeRecord(Code::AccuracyResidual,2672,32,0,
                        EntityKind::Face,f,-1,scale,1,scale,Quantity::Accuracy));
                }
                scale=GPU_OPERATOR_R(1.0);
            }
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
                s.gasSpecies.flux[k*s.nFaces+f]=ugkwp::scaleInventoryFlux(s.gasSpecies.flux[k*s.nFaces+f],scale);
        }
    }
    s.gasPhiRho[f] = ugkwp::scaleInventoryFlux(s.gasPhiRho[f],scale);
    s.gasPhiRhoUx[f] = ugkwp::scaleInventoryFlux(s.gasPhiRhoUx[f],scale);
    s.gasPhiRhoUy[f] = ugkwp::scaleInventoryFlux(s.gasPhiRhoUy[f],scale);
    s.gasPhiRhoUz[f] = ugkwp::scaleInventoryFlux(s.gasPhiRhoUz[f],scale);
    s.gasPhiRhoE[f] = ugkwp::scaleInventoryFlux(s.gasPhiRhoE[f],scale);
    publishLimitedGasWallEnergy(s, f, 0);
}
