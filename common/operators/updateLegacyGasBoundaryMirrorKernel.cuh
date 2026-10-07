#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void updateLegacyGasBoundaryMirrorKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME simulationTime
)
{
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces)
    {
        return;
    }

    if (scheduledInletFaceDevice(s, f))
    {
        const GPU_OPERATOR_REAL temperature = clampMin
        (
            finiteOr(s.scheduledInletTemperature, s.TgasMin),
            s.TgasMin
        );
        const GPU_OPERATOR_REAL pressure = clampMin
        (
            finiteOr(scheduledPressureDevice(s, simulationTime), OfSmall),
            s.rhoMin*s.Rgas*temperature
        );
        const GPU_OPERATOR_REAL density = pressure/clampMin(s.Rgas*temperature, OfSmall);
        s.gasBoundaryP[f] = pressure;
        s.gasBoundaryRho[f] = density;
        s.gasBoundaryT[f] = temperature;
        s.riemannBoundaryP[f] = pressure;
        s.riemannBoundaryRho[f] = density;
        s.riemannBoundaryT[f] = temperature;
        return;
    }

    const int kind = s.gasBoundaryKind[f];
    if (kind == 3 || kind == 4)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int mirrorCell = kind == 5 ? coupledFaceNeighbour(s, f) : own;
    if (mirrorCell < 0 || mirrorCell >= s.nCells)
    {
        return;
    }

    if (s.gasBoundaryRhoFix[f] == 0)
    {
        s.gasBoundaryRho[f] = s.rho[mirrorCell];
    }
    if (s.gasBoundaryUFix[f] == 0)
    {
        s.gasBoundaryUx[f] = s.Ux[mirrorCell];
        s.gasBoundaryUy[f] = s.Uy[mirrorCell];
        s.gasBoundaryUz[f] = s.Uz[mirrorCell];
    }
    if (s.gasBoundaryTFix[f] == 0)
    {
        s.gasBoundaryT[f] = s.Tgas[mirrorCell];
    }
    if (s.gasBoundaryPFix[f] == 0 && s.gasBoundaryPWave[f] == 0)
    {
        s.gasBoundaryP[f] = s.p[mirrorCell];
    }
}

template<class GasState>
__global__ void updateRiemannBoundaryMirrorKernel(GasState* sp)
{
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces)
    {
        return;
    }

    const int kind = s.riemannBoundaryKind[f];
    if (kind == 3 || kind == 4)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int mirrorCell = kind == 5 ? coupledFaceNeighbour(s, f) : own;
    if (mirrorCell < 0 || mirrorCell >= s.nCells)
    {
        return;
    }

    if (s.riemannBoundaryRhoFix[f] == 0)
    {
        s.riemannBoundaryRho[f] = s.rho[mirrorCell];
    }
    if (s.riemannBoundaryUFix[f] == 0)
    {
        s.riemannBoundaryUx[f] = s.Ux[mirrorCell];
        s.riemannBoundaryUy[f] = s.Uy[mirrorCell];
        s.riemannBoundaryUz[f] = s.Uz[mirrorCell];
    }
    if (s.riemannBoundaryTFix[f] == 0)
    {
        s.riemannBoundaryT[f] = s.Tgas[mirrorCell];
    }
    if
    (
        s.riemannBoundaryPFix[f] == 0
     && s.riemannBoundaryPWave[f] == 0
    )
    {
        s.riemannBoundaryP[f] = s.p[mirrorCell];
    }
}

#include "reconstructGasMixtureFace.cuh"

template<class GasState>
__device__ GasPrimDevice reconstructGasCellToFace
(
    const GasState& s,
    const int c,
    const int f
)
{
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if (ugkwp::mixtureGasActive(s))
        {
            GasPrimDevice face=gasCellPrimitive(s,c);
            GPU_OPERATOR_REAL y[ugkwp::GasStateTraits<GasState>::speciesCount];
            if(!reconstructGasMixtureFace(s,c,f,face,y))
                s.gasSpecies.faceStatus[f]=int(ugkwp::GasTransportCode::InvalidThermodynamics);
            return face;
        }
    if (s.gasReconstruction != 1)
    {
        return makeGasPrimDevice
        (
            s.rho[c],
            s.Ux[c],
            s.Uy[c],
            s.Uz[c],
            s.p[c],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }

    GPU_OPERATOR_REAL mappedCx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL mappedCy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL mappedCz = GPU_OPERATOR_R(0.0);
    periodicMappedCellCentre(s, f, c, mappedCx, mappedCy, mappedCz);
    const GPU_OPERATOR_REAL dx = s.faceCx[f] - mappedCx;
    const GPU_OPERATOR_REAL dy = s.faceCy[f] - mappedCy;
    const GPU_OPERATOR_REAL dz = s.faceCz[f] - mappedCz;
    GPU_OPERATOR_REAL rhoIncrement = s.gasGradientLimiterRho[c]
      *(s.gradRhoX[c]*dx + s.gradRhoY[c]*dy + s.gradRhoZ[c]*dz);
    GPU_OPERATOR_REAL pressureIncrement = s.gasGradientLimiterP[c]
      *(s.gradPx[c]*dx + s.gradPy[c]*dy + s.gradPz[c]*dz);
    GPU_OPERATOR_REAL uxIncrement = s.gasGradientLimiterUx[c]
      *(s.gradUxX[c]*dx + s.gradUxY[c]*dy + s.gradUxZ[c]*dz);
    GPU_OPERATOR_REAL uyIncrement = s.gasGradientLimiterUy[c]
      *(s.gradUyX[c]*dx + s.gradUyY[c]*dy + s.gradUyZ[c]*dz);
    GPU_OPERATOR_REAL uzIncrement = s.gasGradientLimiterUz[c]
      *(s.gradUzX[c]*dx + s.gradUzY[c]*dy + s.gradUzZ[c]*dz);

                                                                             
                                                                       
                                                                           
                                                                        
    const GPU_OPERATOR_REAL rho = s.rho[c] + rhoIncrement;
    const GPU_OPERATOR_REAL ux = s.Ux[c] + uxIncrement;
    const GPU_OPERATOR_REAL uy = s.Uy[c] + uyIncrement;
    const GPU_OPERATOR_REAL uz = s.Uz[c] + uzIncrement;
    const GPU_OPERATOR_REAL p = s.p[c] + pressureIncrement;
    return makeGasPrimDevice
    (
        rho, ux, uy, uz, p, s.Rgas, s.rhoMin, s.TgasMin
    );
}

template<class GasState>
__device__ GPU_OPERATOR_REAL molecularGasConductivity(const GasState& s)
{
    if constexpr (ugkwp::GasDirectConductivity<GasState>::value)
        if (ugkwp::gasHasDirectConductivity(s)) return s.gasThermalConductivity;
    return s.gasMu*s.gasCp/s.gasPrClamped;
}
