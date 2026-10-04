// Immutable fa71f477 reference for old/new regression, never production.
__global__ void clearPoissonThermalPoolKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
#ifdef UGKP_DEVELOPMENT_PROBES
    if (c == 0 && s.diagnosticPreTransportParticleCount != nullptr)
    {
        *s.diagnosticPreTransportParticleCount =
            clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    }
#endif
    if (c >= s.nCells)
    {
        return;
    }

    // This producer runs after this step's pressure kick and primitive
    // recovery. Only multi-segment consumers need cross-block reuse.
    if (s.csrHeavyReductionEnabled != 0 && s.csrCellTaskCount[c] > 1)
    {
        s.poissonCellCollisionProbability[c] =
            poissonCollisionProbabilityForCell(s, c, dt);
    }
    s.poolThermalCount[c] = 0;
    s.poolThermalSumUx[c] = 0.0;
    s.poolThermalSumUy[c] = 0.0;
    s.poolThermalSumUz[c] = 0.0;
    s.poolThermalSumU2[c] = 0.0;
    s.poissonPoolSampleTargetCount[c] = 0;
    s.poissonPoolMass[c] = 0.0;
    s.poissonPoolMomX[c] = 0.0;
    s.poissonPoolMomY[c] = 0.0;
    s.poissonPoolMomZ[c] = 0.0;
    s.poissonPoolEnergy[c] = 0.0;
    s.poissonPoolDiameter[c] = 0.0;
    s.poissonPoolDiameter2[c] = 0.0;
}
