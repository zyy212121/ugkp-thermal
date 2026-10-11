#pragma once
#include "GpuCellNeighbour.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasGradientLimiterKernel(GasState* sp)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            GPU_OPERATOR_REAL joint=GPU_OPERATOR_R(1.0);
            const int start=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
            for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
            {
                const int index=k*s.nCells+c;
                const GPU_OPERATOR_REAL yc=s.gasSpecies.rho[index]/s.rho[c];
                GPU_OPERATOR_REAL ymin=yc,ymax=yc;
                for(int i=0;i<count;++i)
                {
                    const int f=s.cellFaceId[start+i];if(f<0||f>=s.nFaces)continue;
                    const int nei=coupledFaceNeighbour(s,f);
                    GPU_OPERATOR_REAL adjacent=yc;
                    if(nei>=0)
                    {
                        const int other=c==s.faceOwner[f]?nei:s.faceOwner[f];
                        adjacent=s.gasSpecies.rho[k*s.nCells+other]/s.rho[other];
                    }
                    else if(s.riemannBoundaryKind[f]==0 && s.gasSpecies.compositionBoundaryFixed[f])
                        adjacent=s.gasSpecies.boundaryMassFraction[k*s.nFaces+f];
                    ymin=fmin(ymin,adjacent);ymax=fmax(ymax,adjacent);
                }
                for(int i=0;i<count;++i)
                {
                    const int f=s.cellFaceId[start+i];if(f<0||f>=s.nFaces)continue;
                    GPU_OPERATOR_REAL cx,cy,cz;periodicMappedCellCentre(s,f,c,cx,cy,cz);
                    const GPU_OPERATOR_REAL predicted=yc+s.gasSpecies.gradX[index]*(s.faceCx[f]-cx)
                        +s.gasSpecies.gradY[index]*(s.faceCy[f]-cy)+s.gasSpecies.gradZ[index]*(s.faceCz[f]-cz);
                    if(s.gasLimiter!=0)updateConfiguredGasLimiter(s.gasLimiter,yc,predicted,ymin,ymax,joint);
                    updateBarthLimiter(yc,predicted,GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0),joint);
                }
            }
            s.gasSpecies.limiter[c]=clampRange(finiteOr(joint,GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
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
    const GasPrimDevice qc = gasCellPrimitive(s,c);
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
            qf = gasCellPrimitive(s,other);
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
