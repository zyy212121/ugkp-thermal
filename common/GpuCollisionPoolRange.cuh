#pragma once
// Common indexed/direct range traversal and eight-component block reduction.
template<bool PoissonMode>
__device__ void accumulateCsrHeavyPoolTask
(
    DeviceState& s,
    const int c,
    const int begin,
    const int end,
    const bool directParticleIndex,
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

    if (!PoissonMode || collisionProbability > GPU_OPERATOR_R(0.0))
    {
        for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x)
        {
            const int i = directParticleIndex
              ? pos
              : s.sortedParticleIndex[pos];
            accumulateOnePoolParticle<PoissonMode>
            (
                s, c, i, collisionProbability,
                mass, momX, momY, momZ, energy, diameter, diameter2, count
            );
        }
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
