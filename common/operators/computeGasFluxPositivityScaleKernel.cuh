#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeGasFluxPositivityScaleKernel(DeviceState* sp, const GPU_OPERATOR_REAL dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL outgoingMassFlux = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const GPU_OPERATOR_REAL phi = finiteOr(s.gasPhiRho[f], 0.0);
        if (s.faceOwner[f] == c && phi > 0.0)
        {
            outgoingMassFlux += phi;
        }
        else if (s.faceNeighbour[f] == c && phi < 0.0)
        {
            outgoingMassFlux -= phi;
        }
    }

    GPU_OPERATOR_REAL scale = 1.0;
    if (outgoingMassFlux > 0.0 && dt > 0.0)
    {
        const GPU_OPERATOR_REAL availableMass =
            clampMin(s.rho[c] - s.rhoMin, 0.0)*s.V[c];
        const GPU_OPERATOR_REAL requestedOutflowMass = dt*outgoingMassFlux;
        if (requestedOutflowMass > availableMass)
        {
            scale = clampRange
            (
                0.999*availableMass
               /(requestedOutflowMass + 1.0e-300),
                0.0,
                1.0
            );
        }
    }
    s.gasFluxPositivityScale[c] = scale;
}

__global__ void applyGasFluxPositivityScaleKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const GPU_OPERATOR_REAL phi = finiteOr(s.gasPhiRho[f], 0.0);
    GPU_OPERATOR_REAL scale = 1.0;
    if (phi > 0.0)
    {
        const int own = s.faceOwner[f];
        if (own >= 0 && own < s.nCells)
        {
            scale = s.gasFluxPositivityScale[own];
        }
    }
    else if (phi < 0.0 && coupledFaceNeighbour(s, f) >= 0)
    {
        const int nei = s.faceNeighbour[f];
        if (nei >= 0 && nei < s.nCells)
        {
            scale = s.gasFluxPositivityScale[nei];
        }
    }

    scale = clampRange(finiteOr(scale, 0.0), 0.0, 1.0);
    s.gasPhiRho[f] *= scale;
    s.gasPhiRhoUx[f] *= scale;
    s.gasPhiRhoUy[f] *= scale;
    s.gasPhiRhoUz[f] *= scale;
    s.gasPhiRhoE[f] *= scale;
}
