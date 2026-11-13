#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class Value, class Low, class High>
__device__ Value clampRange(const Value x, const Low lo, const High hi)
{
    return x < lo ? Value(lo) : (x > hi ? Value(hi) : x);
}
