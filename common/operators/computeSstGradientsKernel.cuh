#pragma once
#include "GpuCellNeighbour.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeSstGradientsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    GPU_OPERATOR_REAL gkx = GPU_OPERATOR_R(0.0), gky = GPU_OPERATOR_R(0.0), gkz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gox = GPU_OPERATOR_R(0.0), goy = GPU_OPERATOR_R(0.0), goz = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        GPU_OPERATOR_REAL kFace = s.k[c];
        GPU_OPERATOR_REAL omegaFace = s.omega[c];
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = oppositeCellAcrossFace(c, own, nei);
            if (other >= 0 && other < s.nCells)
            {
                const GPU_OPERATOR_REAL ownerWeight =
                    clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
                const GPU_OPERATOR_REAL cellWeight =
                    c == own ? ownerWeight : GPU_OPERATOR_R(1.0) - ownerWeight;
                kFace =
                    cellWeight*s.k[c] + (GPU_OPERATOR_R(1.0) - cellWeight)*s.k[other];
                omegaFace =
                    cellWeight*s.omega[c]
                  + (GPU_OPERATOR_R(1.0) - cellWeight)*s.omega[other];
            }
        }
        else
        {
            kFace = sstBoundaryValue(s, f, c, false);
            omegaFace = sstBoundaryValue(s, f, c, true);
        }
        const GPU_OPERATOR_REAL sx = sign*s.Sfx[f];
        const GPU_OPERATOR_REAL sy = sign*s.Sfy[f];
        const GPU_OPERATOR_REAL sz = sign*s.Sfz[f];
        gkx += kFace*sx;
        gky += kFace*sy;
        gkz += kFace*sz;
        gox += omegaFace*sx;
        goy += omegaFace*sy;
        goz += omegaFace*sz;
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    s.gradKX[c] = gkx*invV;
    s.gradKY[c] = gky*invV;
    s.gradKZ[c] = gkz*invV;
    s.gradOmegaX[c] = gox*invV;
    s.gradOmegaY[c] = goy*invV;
    s.gradOmegaZ[c] = goz*invV;
}
