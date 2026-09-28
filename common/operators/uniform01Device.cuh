#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL uniform01Device(unsigned long long& state)
{
    state = mixSeed(state + 0x9e3779b97f4a7c15ULL);
    return static_cast<GPU_OPERATOR_REAL>(state >> 11)*(1.0/9007199254740992.0);
}
