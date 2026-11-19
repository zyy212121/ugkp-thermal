#pragma once



template<bool DeferAcceptedRng>
__device__ __forceinline__ bool selectPoissonPoolParticle
(DeviceState& s, const int i, const GPU_OPERATOR_REAL probability, unsigned long long& rng)
{
    rng = s.pRng[i];
    if (uniform01Device(rng) >= probability)
    {
        s.pRng[i] = rng;
        return false;
    }
    if constexpr (!DeferAcceptedRng) s.pRng[i] = rng;
    return true;
}
