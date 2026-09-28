#pragma once
// Shared split-directory S1 protocol; scalar-tree/component reduction remains a compile-time policy.
#if GPU_POOL_STATIC_DIRECTORY
template<bool DirectParticleIndex, bool AddToExistingPool, bool HeavyReductionEnabled>
__global__ void accumulatePoissonPoolSplitSegmentByCellKernel
#else
template<bool DirectParticleIndex, bool AddToExistingPool>
__global__ void accumulatePoissonPoolSplitSegmentKernel
#endif
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

#if GPU_POOL_STATIC_DIRECTORY
    const int start = DirectParticleIndex
      ? s.preBaseCellOffset[c]
      : s.cellParticleOffset[c];
    const int end = DirectParticleIndex
      ? s.preBaseCellOffset[c + 1]
      : s.cellParticleOffset[c + 1];
    if (start >= end)
    {
        return;
    }

    if constexpr (HeavyReductionEnabled)
    {
        return;
    }

#else
    const int baseBegin = s.preBaseCellOffset[c];
    const int baseEnd = s.preBaseCellOffset[c + 1];
    const int injectionBegin = s.cellParticleOffset[c];
    const int injectionEnd = s.cellParticleOffset[c + 1];
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
        cellCollisionProbability =
            (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
          ? GPU_OPERATOR_R(0.0)
          : clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
#endif
    }

    __syncthreads();

#if GPU_POOL_STATIC_DIRECTORY
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

#else
    const int start = DirectParticleIndex ? baseBegin : injectionBegin;
    const int end = DirectParticleIndex ? baseEnd : injectionEnd;
    GPU_OPERATOR_REAL sums[8] = {GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.0)};
    const GPU_OPERATOR_REAL prob = cellCollisionProbability;
    if (prob > GPU_OPERATOR_R(0.0))
    {
#endif
    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = DirectParticleIndex
          ? pos
          : s.sortedParticleIndex[pos];
        accumulateOnePoolParticle<true, !GPU_POOL_STATIC_DIRECTORY>
        (s, c, i, prob, GPU_SPLIT_POOL_SUM(0, locMass), GPU_SPLIT_POOL_SUM(1, locMomX), GPU_SPLIT_POOL_SUM(2, locMomY), GPU_SPLIT_POOL_SUM(3, locMomZ), GPU_SPLIT_POOL_SUM(4, locEnergy), GPU_SPLIT_POOL_SUM(5, locDiameter), GPU_SPLIT_POOL_SUM(6, locDiameter2), GPU_SPLIT_POOL_SUM(7, locCount));
    }

#if GPU_POOL_STATIC_DIRECTORY
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

#else
    }
    extern __shared__ GPU_OPERATOR_REAL warpPartials[];
    blockReduceComponentSums<8>(sums, warpPartials);
#endif
    if (threadIdx.x == 0)
    {
        if GPU_POOL_CONSTEXPR (AddToExistingPool)
        {
            s.poissonPoolMass[c] += GPU_SPLIT_POOL_RESULT(0, shMass);
            s.poissonPoolMomX[c] += GPU_SPLIT_POOL_RESULT(1, shMomX);
            s.poissonPoolMomY[c] += GPU_SPLIT_POOL_RESULT(2, shMomY);
            s.poissonPoolMomZ[c] += GPU_SPLIT_POOL_RESULT(3, shMomZ);
            s.poissonPoolEnergy[c] += GPU_SPLIT_POOL_RESULT(4, shEnergy);
            s.poissonPoolDiameter[c] += GPU_SPLIT_POOL_RESULT(5, shDiameter);
            s.poissonPoolDiameter2[c] += GPU_SPLIT_POOL_RESULT(6, shDiameter2);
            s.poolThermalCount[c] += static_cast<int>(GPU_SPLIT_POOL_RESULT(7, shCount));
        }
        else
        {
            s.poissonPoolMass[c] = GPU_SPLIT_POOL_RESULT(0, shMass);
            s.poissonPoolMomX[c] = GPU_SPLIT_POOL_RESULT(1, shMomX);
            s.poissonPoolMomY[c] = GPU_SPLIT_POOL_RESULT(2, shMomY);
            s.poissonPoolMomZ[c] = GPU_SPLIT_POOL_RESULT(3, shMomZ);
            s.poissonPoolEnergy[c] = GPU_SPLIT_POOL_RESULT(4, shEnergy);
            s.poissonPoolDiameter[c] = GPU_SPLIT_POOL_RESULT(5, shDiameter);
            s.poissonPoolDiameter2[c] = GPU_SPLIT_POOL_RESULT(6, shDiameter2);
            s.poolThermalCount[c] = static_cast<int>(GPU_SPLIT_POOL_RESULT(7, shCount));
        }
    }
}
