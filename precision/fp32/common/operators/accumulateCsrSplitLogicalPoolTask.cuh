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
    float (&sums)[8],
    float* warpPartials
)
{
    float mass = GPU_OPERATOR_R(0.0);
    float momX = GPU_OPERATOR_R(0.0);
    float momY = GPU_OPERATOR_R(0.0);
    float momZ = GPU_OPERATOR_R(0.0);
    float energy = GPU_OPERATOR_R(0.0);
    float diameter = GPU_OPERATOR_R(0.0);
    float diameter2 = GPU_OPERATOR_R(0.0);
    float count = GPU_OPERATOR_R(0.0);

    const int baseBegin = s.preBaseCellOffset[c];
    const int baseCount = s.preBaseCellOffset[c + 1] - baseBegin;
    const int injectionBegin = s.cellParticleOffset[c];
    const int baseLogicalEnd = min(logicalEnd, baseCount);
    for (unsigned int cursor = static_cast<unsigned int>(logicalBegin) + threadIdx.x;
         cursor < static_cast<unsigned int>(baseLogicalEnd); cursor += blockDim.x)
    {
        const int logical = static_cast<int>(cursor);
        accumulateCsrSplitLogicalPoolParticle<PoissonMode>
        (
            s, c, baseBegin + logical, collisionProbability,
            mass, momX, momY, momZ, energy, diameter, diameter2, count
        );
    }

    const int injectionLogicalBegin = max(logicalBegin, baseCount);
    for (unsigned int cursor = static_cast<unsigned int>(injectionLogicalBegin) + threadIdx.x;
         cursor < static_cast<unsigned int>(logicalEnd); cursor += blockDim.x)
    {
        const int logical = static_cast<int>(cursor);
        const int injectionPosition =
            injectionBegin + (logical - baseCount);
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
    blockReduceCollisionMoments<8>(sums, warpPartials);
}
