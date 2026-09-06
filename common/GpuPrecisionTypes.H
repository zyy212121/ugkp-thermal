#ifndef UGKWP_GPU_PRECISION_TYPES_H
#define UGKWP_GPU_PRECISION_TYPES_H
#include <cfloat>
#ifndef UGKWP_GPU_REAL_BITS
#define UGKWP_GPU_REAL_BITS 64
#endif
using GpuTime = double;
                                                                              
using GpuWallEnergy = double;
static_assert(sizeof(GpuWallEnergy) == 8, "Wall energy accumulation is always binary64");
#if UGKWP_GPU_REAL_BITS == 32
using GpuReal = float;
#define GPU_REAL_MAX FLT_MAX
#define GPU_REAL_MIN FLT_MIN
#define GPU_REAL_EPSILON FLT_EPSILON
#define GPU_REAL_LARGE 1.0e30f
#define GPU_REAL_TINY 1.0e-30f
#define GPU_LARGE(value) GPU_REAL_LARGE
#define GPU_TINY(value) GPU_REAL_TINY
#elif UGKWP_GPU_REAL_BITS == 64
using GpuReal = double;
#define GPU_REAL_MAX DBL_MAX
#define GPU_REAL_MIN DBL_MIN
#define GPU_REAL_EPSILON DBL_EPSILON
#define GPU_REAL_LARGE 1.0e300
#define GPU_REAL_TINY 1.0e-300
#define GPU_LARGE(value) value
#define GPU_TINY(value) value
#else
#error UGKWP_GPU_REAL_BITS must be 32 or 64
#endif
#define GPU_R(value) static_cast<GpuReal>(value)
static_assert(sizeof(GpuTime) == 8, "Time must stay binary64");
static_assert(sizeof(GpuReal)*8 == UGKWP_GPU_REAL_BITS, "Physical precision mismatch");
#endif
