#pragma once
#include "GpuWallEnergySink.cuh"
#include "gasTransport/GasGeometryValidation.H"

// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasFluxPositivityScaleKernel(GasState* sp, const GPU_OPERATOR_TIME dt)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL inventoryVolume=s.V[c];
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL newVolume;
        if(!ugkwp::gasGeometryCellVolumes(s,c,dt,inventoryVolume,newVolume))
        {if(!ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");s.gasFluxPositivityScale[c]=GPU_OPERATOR_R(0.0);return;}
    }
    GPU_OPERATOR_REAL outgoingMassFlux = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
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
        const GPU_OPERATOR_REAL availableMass =
            clampMin(s.rho[c] - s.rhoMin, GPU_OPERATOR_R(0.0))*inventoryVolume;
        const GPU_OPERATOR_REAL requestedOutflowMass = dt*outgoingMassFlux;
        if (requestedOutflowMass > availableMass)
        {
            scale = clampRange
            (
                GPU_OPERATOR_R(0.999)*availableMass
               /(requestedOutflowMass + GPU_OPERATOR_TINY(1.0e-300)),
                GPU_OPERATOR_R(0.0),
                GPU_OPERATOR_R(1.0)
            );
        }
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
                    if (!ugkwp::gasFinite(flux))
                    { s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::NonFiniteState); return; }
                    if (s.faceOwner[f]==c && flux>GPU_OPERATOR_R(0.0)) outflow+=flux;
                    else if (s.faceNeighbour[f]==c && flux<GPU_OPERATOR_R(0.0)) outflow-=flux;
                }
                const GPU_OPERATOR_REAL available=s.gasSpecies.rho[k*s.nCells+c]*inventoryVolume;
                GPU_OPERATOR_REAL factor=GPU_OPERATOR_R(1.0);
                if (dt>GPU_OPERATOR_R(0.0) && outflow>GPU_OPERATOR_R(0.0)
                    && dt*outflow>available)
                    factor=clampRange(GPU_OPERATOR_R(0.999)*available/(dt*outflow),GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
                s.gasSpecies.positivityScale[k*s.nCells+c]=factor;
            }
    }
}

template<class GasState>
__global__ void applyGasFluxPositivityScaleKernel(GasState* sp)
{
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
                const int donor=flux>GPU_OPERATOR_R(0.0)?s.faceOwner[f]:coupledFaceNeighbour(s,f);
                if (flux!=GPU_OPERATOR_R(0.0) && donor>=0 && donor<s.nCells)
                    scale=fmin(scale,s.gasSpecies.positivityScale[k*s.nCells+donor]);
            }
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
                s.gasSpecies.flux[k*s.nFaces+f]*=scale;
        }
    }
    s.gasPhiRho[f] *= scale;
    s.gasPhiRhoUx[f] *= scale;
    s.gasPhiRhoUy[f] *= scale;
    s.gasPhiRhoUz[f] *= scale;
    s.gasPhiRhoE[f] *= scale;
    publishLimitedGasWallEnergy(s, f, 0);
}
