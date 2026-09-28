#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computePressureGradientKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL gx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gz = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const int f = s.cellFaceId[p];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const int kind = s.cellFaceKind[p];
        const int own = s.faceOwner[f];
        const int neiFace = s.faceNeighbour[f];
        const GPU_OPERATOR_REAL lambda = finiteOr(s.faceWeight[f], GPU_OPERATOR_R(0.5));
        const GPU_OPERATOR_REAL pf =
            (own >= 0 && own < s.nCells && neiFace >= 0 && neiFace < s.nCells)
          ? lambda*s.p[own] + (GPU_OPERATOR_R(1.0) - lambda)*s.p[neiFace]
          : ((kind != 4 && f >= 0 && f < s.nFaces)
              ? s.gasBoundaryP[f]
              : s.p[c]);
        const GPU_OPERATOR_REAL sign = (own == c) ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        gx += pf*sign*s.Sfx[f];
        gy += pf*sign*s.Sfy[f];
        gz += pf*sign*s.Sfz[f];
    }

    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], s.rhoMin);
    s.gradPx[c] = gx*invV;
    s.gradPy[c] = gy*invV;
    s.gradPz[c] = gz*invV;
}
