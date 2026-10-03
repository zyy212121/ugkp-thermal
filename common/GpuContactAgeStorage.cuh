#pragma once
template<class State>
__device__ __forceinline__ auto particleContactAgeStorage(State& s, const int i, int)
    -> decltype((s.pContactAge[i]))
{
    return s.pContactAge[i];
}
template<class State>
__device__ __forceinline__ auto particleContactAgeStorage(State& s, const int i, long)
    -> decltype((s.pTheta[i]))
{
    return s.pTheta[i];
}
template<class State>
__device__ __forceinline__ auto resetSeparateContactAge(State& s, const int i, int)
    -> decltype(s.pContactAge[i] = 0, void())
{
    s.pContactAge[i] = 0;
}
template<class State>
__device__ __forceinline__ void resetSeparateContactAge(State&, const int, long) {}
