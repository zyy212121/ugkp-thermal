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

    if (!PoissonMode || collisionProbability > GPU_OPERATOR_R(0.0))
    {
        for (unsigned int cursor = static_cast<unsigned int>(begin) + threadIdx.x;
         cursor < static_cast<unsigned int>(end); cursor += blockDim.x)
    {
        const int pos = static_cast<int>(cursor);
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
    blockReduceCollisionMoments<8>(sums, warpPartials);
}
