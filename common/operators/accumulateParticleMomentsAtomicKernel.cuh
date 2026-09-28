#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void accumulateParticleMomentsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const GPU_OPERATOR_REAL m = clampMin(finiteOr(s.pm[i], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL ux = finiteOr(s.pux[i], GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL uy = finiteOr(s.puy[i], GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL uz = finiteOr(s.puz[i], GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL theta = particleMomentThetaDevice(s, i);
        const GPU_OPERATOR_REAL d =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                GPU_OPERATOR_R(1.0e-12)
            );
        const GPU_OPERATOR_REAL tp =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        atomicAdd(&s.momRhoP[c], m);
        atomicAdd(&s.momRhoUPx[c], m*ux);
        atomicAdd(&s.momRhoUPy[c], m*uy);
        atomicAdd(&s.momRhoUPz[c], m*uz);
        atomicAdd
        (
            &s.momRhoEP[c],
            m*(GPU_OPERATOR_R(0.5)*sqr3(ux, uy, uz) + GPU_OPERATOR_R(1.5)*theta)
        );
        atomicAdd(&s.momRhoPD[c], m*d);
        atomicAdd(&s.momRhoHpP[c], m*particleSpecificEnthalpyDevice(tp));
        atomicAdd(&s.cellParticleCount[c], 1);
    }
}
