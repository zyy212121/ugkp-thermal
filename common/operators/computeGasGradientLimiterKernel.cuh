#pragma once
#include "GpuCellNeighbour.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeGasGradientLimiterKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if (s.gasLimiter == 0)
    {
        s.gasGradientLimiterRho[c] = GPU_OPERATOR_R(1.0);
        s.gasGradientLimiterUx[c] = GPU_OPERATOR_R(1.0);
        s.gasGradientLimiterUy[c] = GPU_OPERATOR_R(1.0);
        s.gasGradientLimiterUz[c] = GPU_OPERATOR_R(1.0);
        s.gasGradientLimiterP[c] = GPU_OPERATOR_R(1.0);
        s.gasGradientLimiterT[c] = GPU_OPERATOR_R(1.0);
        return;
    }
    const GasPrimDevice qc = makeGasPrimDevice
    (
        s.rho[c], s.Ux[c], s.Uy[c], s.Uz[c], s.p[c],
        s.Rgas, s.rhoMin, s.TgasMin
    );
    GPU_OPERATOR_REAL rMin = qc.rho, rMax = qc.rho;
    GPU_OPERATOR_REAL uxMin = qc.ux, uxMax = qc.ux;
    GPU_OPERATOR_REAL uyMin = qc.uy, uyMax = qc.uy;
    GPU_OPERATOR_REAL uzMin = qc.uz, uzMax = qc.uz;
    GPU_OPERATOR_REAL pMin = qc.p, pMax = qc.p;
    GPU_OPERATOR_REAL tMin = qc.T, tMax = qc.T;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        GasPrimDevice qf;
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = oppositeCellAcrossFace(c, own, nei);
            qf = makeGasPrimDevice
            (
                s.rho[other],
                s.Ux[other],
                s.Uy[other],
                s.Uz[other],
                s.p[other],
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
            qf.T = clampMin
            (
                finiteOr(s.Tgas[other], qf.T),
                s.TgasMin
            );
        }
        else
        {
            qf = riemannFacePrimitiveForGradient(s, c, f);
        }
        rMin = fmin(rMin, qf.rho); rMax = fmax(rMax, qf.rho);
        uxMin = fmin(uxMin, qf.ux); uxMax = fmax(uxMax, qf.ux);
        uyMin = fmin(uyMin, qf.uy); uyMax = fmax(uyMax, qf.uy);
        uzMin = fmin(uzMin, qf.uz); uzMax = fmax(uzMax, qf.uz);
        pMin = fmin(pMin, qf.p); pMax = fmax(pMax, qf.p);
        tMin = fmin(tMin, qf.T); tMax = fmax(tMax, qf.T);
    }
    rMin = fmax(rMin, s.rhoMin);
    pMin = fmax(pMin, s.rhoMin);
    tMin = fmax(tMin, s.TgasMin);
    GPU_OPERATOR_REAL rhoLimiter = GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL uxLimiter = GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL uyLimiter = GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL uzLimiter = GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL pLimiter = GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL tLimiter = GPU_OPERATOR_R(1.0);
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        const GPU_OPERATOR_REAL dx = s.faceCx[f] - s.Cx[c];
        const GPU_OPERATOR_REAL dy = s.faceCy[f] - s.Cy[c];
        const GPU_OPERATOR_REAL dz = s.faceCz[f] - s.Cz[c];
        updateConfiguredGasLimiter(s.gasLimiter, qc.rho, qc.rho + s.gradRhoX[c]*dx + s.gradRhoY[c]*dy + s.gradRhoZ[c]*dz, rMin, rMax, rhoLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.ux, qc.ux + s.gradUxX[c]*dx + s.gradUxY[c]*dy + s.gradUxZ[c]*dz, uxMin, uxMax, uxLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.uy, qc.uy + s.gradUyX[c]*dx + s.gradUyY[c]*dy + s.gradUyZ[c]*dz, uyMin, uyMax, uyLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.uz, qc.uz + s.gradUzX[c]*dx + s.gradUzY[c]*dy + s.gradUzZ[c]*dz, uzMin, uzMax, uzLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.p, qc.p + s.gradPx[c]*dx + s.gradPy[c]*dy + s.gradPz[c]*dz, pMin, pMax, pLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.T, qc.T + s.gradTX[c]*dx + s.gradTY[c]*dy + s.gradTZ[c]*dz, tMin, tMax, tLimiter);
    }
    s.gasGradientLimiterRho[c] =
        clampRange(finiteOr(rhoLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.gasGradientLimiterUx[c] =
        clampRange(finiteOr(uxLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.gasGradientLimiterUy[c] =
        clampRange(finiteOr(uyLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.gasGradientLimiterUz[c] =
        clampRange(finiteOr(uzLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.gasGradientLimiterP[c] =
        clampRange(finiteOr(pLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.gasGradientLimiterT[c] =
        clampRange(finiteOr(tLimiter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
}
