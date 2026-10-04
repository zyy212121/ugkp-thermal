#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
#if GPU_POOL_INITIALIZATION_WITH_PROBABILITY
__global__ void clearPoissonThermalPoolKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
#else
__global__ void clearPoissonThermalPoolKernel(DeviceState* sp)
#endif
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
#if GPU_POOL_INITIALIZATION_WITH_PROBABILITY
#ifdef UGKP_DEVELOPMENT_PROBES
    if (c == 0 && s.diagnosticPreTransportParticleCount != nullptr)
    {
        *s.diagnosticPreTransportParticleCount =
            clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    }
#endif
#endif
    if (c >= s.nCells)
    {
        return;
    }

#if GPU_POOL_INITIALIZATION_WITH_PROBABILITY
    // Preserve pressure kick -> primitive recovery -> probability publication.
    if (s.csrHeavyReductionEnabled != 0 && s.csrCellTaskCount[c] > 1)
    {
        s.poissonCellCollisionProbability[c] =
            poissonCollisionProbabilityForCell(s, c, dt);
    }
#endif
    s.poolThermalCount[c] = 0;
    s.poolThermalSumUx[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumUy[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumUz[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumU2[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolSampleTargetCount[c] = 0;
    s.poissonPoolMass[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomX[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomY[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomZ[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolEnergy[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolDiameter[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolDiameter2[c] = GPU_OPERATOR_R(0.0);
}
