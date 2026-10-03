#pragma once
// Shuffle in input precision; each face/channel ledger accumulates in double.
__device__ void atomicAddParticleWallEnergyByFace
(
    DeviceState& s,
    GpuWallEnergy* const wallEnergyLedger,
    const int globalFaceId,
    const GPU_OPERATOR_REAL wallEnergyJ
)
{
    if (wallEnergyJ == GPU_OPERATOR_R(0.0))
    {
        return;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned int active = __activemask();
    const int ledgerChannel =
        wallEnergyLedger == s.particleWallDepositedEnergy ? 0 : 1;
    const int faceChannelKey = 2*globalFaceId + ledgerChannel;
    const unsigned int group =
        __match_any_sync(active, faceChannelKey);
    const int leader = __ffs(static_cast<int>(group)) - 1;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    GpuWallEnergy groupEnergy = 0.0;
    unsigned int remaining = group;
    while (remaining != 0u)
    {
        const int sourceLane = __ffs(static_cast<int>(remaining)) - 1;
        groupEnergy += static_cast<GpuWallEnergy>(__shfl_sync(group, wallEnergyJ, sourceLane));
        remaining &= remaining - 1u;
    }
    if (lane == leader)
    {
        atomicAdd(&wallEnergyLedger[globalFaceId], groupEnergy);
    }
#else
    atomicAdd(&wallEnergyLedger[globalFaceId], static_cast<GpuWallEnergy>(wallEnergyJ));
#endif
}
