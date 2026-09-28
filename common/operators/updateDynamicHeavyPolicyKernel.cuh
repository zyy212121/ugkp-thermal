#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void updateDynamicHeavyPolicyKernel
(
    DeviceState* sp,
    const int splitPreDirectoryActive
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    long long population = s.cellParticleOffset[s.nCells];
    if (splitPreDirectoryActive != 0)
    {
        population += *s.preBaseParticleCountDevice;
    }
    const long long concurrency =
        static_cast<long long>(s.reductionBlockThreads)
       *static_cast<long long>(s.multiprocessorCount)
       *static_cast<long long>(s.lightResidentBlocksPerSm);
    if (population < 0 || population > s.particleCapacity || concurrency <= 0)
    {
        asm("trap;");
    }
    const long long total = population > 0 ? population : 1;
    long long shares = (total + concurrency - 1)/concurrency;
    shares = shares > 0 ? shares : 1;
    const long long threshold =
        static_cast<long long>(s.reductionBlockThreads)*shares;
    if (threshold <= 0 || threshold > 2147483647LL)
    {
        asm("trap;");
    }
    s.dynamicHeavyThreshold = static_cast<int>(threshold);
    s.csrHeavyTileParticles = static_cast<int>(threshold);
}
