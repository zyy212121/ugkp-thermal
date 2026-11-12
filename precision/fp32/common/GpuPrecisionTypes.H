#ifndef UGKWP_GPU_PRECISION_TYPES_H
#define UGKWP_GPU_PRECISION_TYPES_H
#include <cfloat>
using GpuTime = double;
using GpuGeometry = double;
using GpuReal = float;
using GpuEnergy = GpuReal;
using GpuWallEnergy = GpuReal;
#define GPU_REAL_MAX FLT_MAX
#define GPU_REAL_MIN FLT_MIN
#define GPU_REAL_EPSILON FLT_EPSILON
#define GPU_REAL_LARGE 1.0e30f
#define GPU_REAL_TINY 1.0e-30f
#define GPU_LARGE(value) GPU_REAL_LARGE
#define GPU_TINY(value) ((value) < 1.0e-200 ? GPU_REAL_TINY : 1.0842021724855044e-19f)
#define GPU_R(value) static_cast<GpuReal>(value)
static_assert(sizeof(GpuTime) == 8, "Time must stay binary64");
#endif
