#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void snapshotParticleGasCouplingStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.couplingRhoOld[c] =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    s.couplingUxOld[c] = finiteOr(s.Ux[c], GPU_OPERATOR_R(0.0));
    s.couplingUyOld[c] = finiteOr(s.Uy[c], GPU_OPERATOR_R(0.0));
    s.couplingUzOld[c] = finiteOr(s.Uz[c], GPU_OPERATOR_R(0.0));
    s.couplingTgasOld[c] =
        clampMin(finiteOr(s.Tgas[c], s.TgasMin), s.TgasMin);
    s.couplingGasViscosity[c] = ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]);
    if (s.solveParticleTemperature && s.particleGasHeatTransferModelId)
    {
        s.couplingGasConductivity[c] = molecularGasConductivity(s,c,s.couplingTgasOld[c]);
        s.couplingGasPrandtl[c] = ugkwp::gasCellMolecularPrandtl(s,c);
        s.couplingGasPrandtlOneThird[c] = ugkwp::gasCellPrandtlOneThird(s,c);
    }
}

__device__ unsigned long long mixSeed(unsigned long long x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
