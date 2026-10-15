#ifndef UGKWP_GPU_PRECISION_TYPES_H
#define UGKWP_GPU_PRECISION_TYPES_H
#include <cfloat>
using GpuTime = double;
using GpuWallEnergy = double;
static_assert(sizeof(GpuWallEnergy) == 8, "Wall energy accumulation is always binary64");
using GpuReal = double;
#define GPU_REAL_MAX DBL_MAX
#define GPU_REAL_MIN DBL_MIN
#define GPU_REAL_EPSILON DBL_EPSILON
#define GPU_REAL_LARGE 1.0e300
#define GPU_REAL_TINY 1.0e-300
#define GPU_LARGE(value) value
#define GPU_TINY(value) value
#define GPU_R(value) static_cast<GpuReal>(value)
static_assert(sizeof(GpuTime) == 8, "Time must stay binary64");
#endif
