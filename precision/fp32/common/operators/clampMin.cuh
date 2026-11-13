#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class Value, class Bound>
__device__ Value clampMin(const Value x, const Bound lo)
{
    return x < lo ? Value(lo) : x;
}

template<class Value>
__device__ bool finiteDevice(const Value x)
{
    return (x == x) && (fabs(x) < 1.0e300);
}

template<class Value>
__device__ bool nonFiniteDevice(const Value x)
{
    return !finiteDevice(x);
}

template<class Value, class Fallback>
__device__ Value finiteOr(const Value x, const Fallback fallback)
{
    return finiteDevice(x) ? x : Value(fallback);
}
