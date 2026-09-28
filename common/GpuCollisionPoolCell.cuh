#pragma once
// Shared ordinary full-directory collision reduction.
#if GPU_POOL_STATIC_DIRECTORY
template<bool HeavyReductionEnabled>
#endif
__global__ void accumulatePoissonPoolParticlesByCellKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;

    if (c >= s.nCells)
    {
        return;
    }

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
#if GPU_POOL_STATIC_DIRECTORY
    if (start >= end) return;
    if constexpr (HeavyReductionEnabled)
    {
        return;
    }
#else
    if (s.csrHeavyReductionEnabled != 0)
    {
        return;
    }

#endif
    __shared__ GPU_OPERATOR_REAL cellCollisionProbability;

    if (threadIdx.x == 0)
    {
#if GPU_POOL_STATIC_DIRECTORY
        cellCollisionProbability = poissonCollisionProbabilityForCell(s, c, dt);
#else
        const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);

        if (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
        {
            cellCollisionProbability = GPU_OPERATOR_R(0.0);
        }
        else
        {
            cellCollisionProbability =
                clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        }
#endif
    }

    __syncthreads();

    const GPU_OPERATOR_REAL prob = cellCollisionProbability;

    if (prob <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    GPU_OPERATOR_REAL locMass = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locMomX = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locMomY = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locMomZ = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locEnergy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locDiameter = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locDiameter2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL locCount = GPU_OPERATOR_R(0.0);

    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];

#if GPU_POOL_STATIC_DIRECTORY
        accumulateOnePoissonPoolParticle
        (
            s, i, c, prob, locMass, locMomX, locMomY, locMomZ,
            locEnergy, locDiameter, locDiameter2, locCount
        );
#else
        accumulateOnePoolParticle<true, true>(s, c, i, prob, locMass, locMomX, locMomY, locMomZ, locEnergy, locDiameter, locDiameter2, locCount);
#endif

    }

    extern __shared__ GPU_OPERATOR_REAL sh[];

    GPU_OPERATOR_REAL* shMass = sh;
    GPU_OPERATOR_REAL* shMomX = shMass + blockDim.x;
    GPU_OPERATOR_REAL* shMomY = shMomX + blockDim.x;
    GPU_OPERATOR_REAL* shMomZ = shMomY + blockDim.x;
    GPU_OPERATOR_REAL* shEnergy = shMomZ + blockDim.x;
    GPU_OPERATOR_REAL* shDiameter = shEnergy + blockDim.x;
    GPU_OPERATOR_REAL* shDiameter2 = shDiameter + blockDim.x;
    GPU_OPERATOR_REAL* shCount = shDiameter2 + blockDim.x;

    shMass[threadIdx.x] = locMass;
    shMomX[threadIdx.x] = locMomX;
    shMomY[threadIdx.x] = locMomY;
    shMomZ[threadIdx.x] = locMomZ;
    shEnergy[threadIdx.x] = locEnergy;
    shDiameter[threadIdx.x] = locDiameter;
    shDiameter2[threadIdx.x] = locDiameter2;
    shCount[threadIdx.x] = locCount;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shMass[threadIdx.x] += shMass[threadIdx.x + stride];
            shMomX[threadIdx.x] += shMomX[threadIdx.x + stride];
            shMomY[threadIdx.x] += shMomY[threadIdx.x + stride];
            shMomZ[threadIdx.x] += shMomZ[threadIdx.x + stride];
            shEnergy[threadIdx.x] += shEnergy[threadIdx.x + stride];
            shDiameter[threadIdx.x] += shDiameter[threadIdx.x + stride];
            shDiameter2[threadIdx.x] += shDiameter2[threadIdx.x + stride];
            shCount[threadIdx.x] += shCount[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        s.poissonPoolMass[c] = shMass[0];
        s.poissonPoolMomX[c] = shMomX[0];
        s.poissonPoolMomY[c] = shMomY[0];
        s.poissonPoolMomZ[c] = shMomZ[0];
        s.poissonPoolEnergy[c] = shEnergy[0];
        s.poissonPoolDiameter[c] = shDiameter[0];
        s.poissonPoolDiameter2[c] = shDiameter2[0];
        s.poolThermalCount[c] = static_cast<int>(shCount[0]);

    }
}
