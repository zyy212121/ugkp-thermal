// Immutable fa71f477 reference for old/new regression, never production.
__global__ void clearPoissonThermalPoolKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

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
