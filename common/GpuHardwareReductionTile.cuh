#pragma once
#if defined(__CUDACC__)
#define GPU_TILE_HD __host__ __device__ __forceinline__
#else
#define GPU_TILE_HD inline
#endif
// Zero denotes invalid input or an unrepresentable tile, never an L2 decision.
GPU_TILE_HD int hardwareReductionTile(const long long population, const int blockThreads,
    const int multiprocessors, const int residentBlocksPerSm)
{
    if (population < 0 || blockThreads <= 0 || multiprocessors <= 0 || residentBlocksPerSm <= 0)
        return 0;
    const long long blocks = static_cast<long long>(multiprocessors)*residentBlocksPerSm;
#if defined(__CUDA_ARCH__)
    const unsigned long long unsignedBlocks = static_cast<unsigned long long>(blocks);
    const unsigned long long unsignedThreads = static_cast<unsigned long long>(blockThreads);
    const unsigned long long low = unsignedBlocks*unsignedThreads;
    if (__umul64hi(unsignedBlocks, unsignedThreads) != 0 || low > 9223372036854775807ULL)
        return 0;
    const long long concurrency = static_cast<long long>(low);
#else
    if (blocks > 9223372036854775807LL/blockThreads) return 0;
    const long long concurrency = blocks*blockThreads;
#endif
    const long long shares = population == 0 ? 1 : 1 + (population - 1)/concurrency;
    if (shares > 2147483647LL) return 0;
    const long long tile = shares*blockThreads;
    return tile > 2147483647LL ? 0 : static_cast<int>(tile);
}
#undef GPU_TILE_HD
