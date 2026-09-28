#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool PoissonMode>
__device__ void accumulateCsrSplitLogicalPoolTask
(
    DeviceState& s,
    const int c,
    const int logicalBegin,
    const int logicalEnd,
    const GPU_OPERATOR_REAL collisionProbability,
    GPU_OPERATOR_REAL (&sums)[8],
    GPU_OPERATOR_REAL* warpPartials
)
{
    GPU_OPERATOR_REAL mass = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momX = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momY = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momZ = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL energy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL diameter = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL diameter2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL count = GPU_OPERATOR_R(0.0);

    const int baseBegin = s.preBaseCellOffset[c];
    const int baseCount = s.preBaseCellOffset[c + 1] - baseBegin;
    const int injectionBegin = s.cellParticleOffset[c];
    const int baseLogicalEnd = min(logicalEnd, baseCount);
    for
    (
        int logical = logicalBegin + threadIdx.x;
        logical < baseLogicalEnd;
        logical += blockDim.x
    )
    {
        accumulateCsrSplitLogicalPoolParticle<PoissonMode>
        (
            s, c, baseBegin + logical, collisionProbability,
            mass, momX, momY, momZ, energy, diameter, diameter2, count
        );
    }

    const int injectionLogicalBegin = max(logicalBegin, baseCount);
    for
    (
        int logical = injectionLogicalBegin + threadIdx.x;
        logical < logicalEnd;
        logical += blockDim.x
    )
    {
        const int injectionPosition =
            injectionBegin + logical - baseCount;
        accumulateCsrSplitLogicalPoolParticle<PoissonMode>
        (
            s, c, s.sortedParticleIndex[injectionPosition],
            collisionProbability,
            mass, momX, momY, momZ, energy, diameter, diameter2, count
        );
    }

    sums[0] = mass;
    sums[1] = momX;
    sums[2] = momY;
    sums[3] = momZ;
    sums[4] = energy;
    sums[5] = diameter;
    sums[6] = diameter2;
    sums[7] = count;
    blockReduceComponentSums<8>(sums, warpPartials);
}
