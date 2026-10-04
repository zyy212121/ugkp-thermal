#pragma once
// Contact-index publication is separate from sampling target preparation.
__device__ void appendSelectedStuckParticleIndex
(
    DeviceState& s,
    const int i
)
{
    if (s.pStuck[i] == 0)
    {
        return;
    }

    const int slot = atomicAdd(s.compactCountDevice, 1);
    if (slot < 0 || slot >= s.particleCapacity)
    {
        asm("trap;");
    }
    s.compactPStatus[slot] = i;
}
