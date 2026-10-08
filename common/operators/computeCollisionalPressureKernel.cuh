#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeCollisionalPressureKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.collisionalPressure[c] = solidPressureFromMomentsDevice(s, c);
}

__global__ void computeCollisionalPressureFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    GPU_OPERATOR_REAL pFace = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL ufx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL ufy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL ufz = GPU_OPERATOR_R(0.0);

    if (own >= 0 && own < s.nCells && nei >= 0 && nei < s.nCells)
    {
        const GPU_OPERATOR_REAL w = clampRange(finiteOr(s.faceWeight[f], GPU_OPERATOR_R(0.5)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        const GPU_OPERATOR_REAL rhoOwn =
            clampMin(finiteOr(s.momRhoP[own], GPU_OPERATOR_R(0.0)), s.epsSMin*s.rhoSolid);
        const GPU_OPERATOR_REAL rhoNei =
            clampMin(finiteOr(s.momRhoP[nei], GPU_OPERATOR_R(0.0)), s.epsSMin*s.rhoSolid);
        pFace =
            w*solidPressureFromMomentsDevice(s, own)
          + (GPU_OPERATOR_R(1.0) - w)*solidPressureFromMomentsDevice(s, nei);
        ufx =
            w*finiteOr(s.momRhoUPx[own], GPU_OPERATOR_R(0.0))/rhoOwn
          + (GPU_OPERATOR_R(1.0) - w)*finiteOr(s.momRhoUPx[nei], GPU_OPERATOR_R(0.0))/rhoNei;
        ufy =
            w*finiteOr(s.momRhoUPy[own], GPU_OPERATOR_R(0.0))/rhoOwn
          + (GPU_OPERATOR_R(1.0) - w)*finiteOr(s.momRhoUPy[nei], GPU_OPERATOR_R(0.0))/rhoNei;
        ufz =
            w*finiteOr(s.momRhoUPz[own], GPU_OPERATOR_R(0.0))/rhoOwn
          + (GPU_OPERATOR_R(1.0) - w)*finiteOr(s.momRhoUPz[nei], GPU_OPERATOR_R(0.0))/rhoNei;
    }
    else if
    (
        own >= 0
     && own < s.nCells
     && (s.gasBoundaryKind[f] == 1 || s.gasBoundaryKind[f] == 2)
    )
    {
        pFace = solidPressureFromMomentsDevice(s, own);
                                                                                 
        ufx = GPU_OPERATOR_R(0.0);
        ufy = GPU_OPERATOR_R(0.0);
        ufz = GPU_OPERATOR_R(0.0);
    }

    const GPU_OPERATOR_REAL fx = pFace*s.Sfx[f];
    const GPU_OPERATOR_REAL fy = pFace*s.Sfy[f];
    const GPU_OPERATOR_REAL fz = pFace*s.Sfz[f];
    s.solidPressurePhiMomX[f] = finiteOr(fx, GPU_OPERATOR_R(0.0));
    s.solidPressurePhiMomY[f] = finiteOr(fy, GPU_OPERATOR_R(0.0));
    s.solidPressurePhiMomZ[f] = finiteOr(fz, GPU_OPERATOR_R(0.0));
    s.solidPressurePhiEnergy[f] =
        finiteOr(ufx*fx + ufy*fy + ufz*fz, GPU_OPERATOR_R(0.0));
}
