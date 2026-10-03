#pragma once
#include <type_traits>
// Required entry capabilities, precision and execution policies. Include after precision aliases
// and before any common operator. No defaults: an omitted flag must never silently select zero.
#ifndef GPU_OPERATOR_THERMAL
#error "missing required entry policy GPU_OPERATOR_THERMAL"
#endif
#ifndef GPU_OPERATOR_REAL
#error "missing required entry policy GPU_OPERATOR_REAL"
#endif
#ifndef GPU_OPERATOR_TIME
#error "missing required entry policy GPU_OPERATOR_TIME"
#endif
#ifndef GPU_OPERATOR_R
#error "missing required entry policy GPU_OPERATOR_R"
#endif
#ifndef GPU_OPERATOR_TINY
#error "missing required entry policy GPU_OPERATOR_TINY"
#endif
#ifndef GPU_POOL_THETA_AFTER_REJECTION
#error "missing required entry policy GPU_POOL_THETA_AFTER_REJECTION"
#endif
#ifndef GPU_POOL_PARTICLE_THETA
#error "missing required entry policy GPU_POOL_PARTICLE_THETA"
#endif
#ifndef GPU_POOL_MASS_FALLBACK
#error "missing required entry policy GPU_POOL_MASS_FALLBACK"
#endif
#ifndef GPU_POOL_SPLIT_REDUCTION_TOPOLOGY
#error "missing required entry policy GPU_POOL_SPLIT_REDUCTION_TOPOLOGY"
#endif
#ifndef GPU_POOL_STATIC_DIRECTORY
#error "missing required entry policy GPU_POOL_STATIC_DIRECTORY"
#endif
#ifndef GPU_POOL_CACHED_PROBABILITY
#error "missing required entry policy GPU_POOL_CACHED_PROBABILITY"
#endif
#ifndef GPU_POOL_SPLIT_DIRECTORY
#error "missing required entry policy GPU_POOL_SPLIT_DIRECTORY"
#endif
#ifndef GPU_POOL_DIRECT_PARTICLE_INDEX
#error "missing required entry policy GPU_POOL_DIRECT_PARTICLE_INDEX"
#endif
#ifndef GPU_POOL_SAMPLING_READY
#error "missing required entry policy GPU_POOL_SAMPLING_READY"
#endif
#ifndef GPU_POOL_END_TASK
#error "missing required entry policy GPU_POOL_END_TASK"
#endif
#ifndef GPU_DIRECTORY_PARAMETER_TYPE
#error "missing required entry policy GPU_DIRECTORY_PARAMETER_TYPE"
#endif
#ifndef GPU_DIRECTORY_ARGUMENT
#error "missing required entry policy GPU_DIRECTORY_ARGUMENT"
#endif
#ifndef GPU_DIRECTORY_HAS_BASE_ONLY
#error "missing required entry policy GPU_DIRECTORY_HAS_BASE_ONLY"
#endif
#ifndef GPU_DIRECTORY_OWNS_TILE_POLICY
#error "missing required entry policy GPU_DIRECTORY_OWNS_TILE_POLICY"
#endif
#ifndef GPU_DIRECTORY_SELECTOR
#error "missing required entry policy GPU_DIRECTORY_SELECTOR"
#endif
#ifndef GPU_DIRECTORY_FULL
#error "missing required entry policy GPU_DIRECTORY_FULL"
#endif
#ifndef GPU_RECOVERY_INLINE
#error "missing required entry policy GPU_RECOVERY_INLINE"
#endif
#ifndef GPU_PERIODIC_FACE_KIND
#error "missing required entry policy GPU_PERIODIC_FACE_KIND"
#endif
#ifndef GPU_PERIODIC_COORDINATE
#error "missing required entry policy GPU_PERIODIC_COORDINATE"
#endif
#ifndef GPU_WALL_COORDINATE
#error "missing required entry policy GPU_WALL_COORDINATE"
#endif
#ifndef GPU_RESET_CONTACT_AGE
#error "missing required entry policy GPU_RESET_CONTACT_AGE"
#endif
#ifndef GPU_CONTACT_DURATION
#error "missing required entry policy GPU_CONTACT_DURATION"
#endif
#ifndef GPU_CONTACT_PEAK
#error "missing required entry policy GPU_CONTACT_PEAK"
#endif
#if GPU_OPERATOR_THERMAL != 0 && GPU_OPERATOR_THERMAL != 1
#error "GPU_OPERATOR_THERMAL must be 0 or 1"
#endif
#if GPU_POOL_THETA_AFTER_REJECTION != 0 && GPU_POOL_THETA_AFTER_REJECTION != 1
#error "GPU_POOL_THETA_AFTER_REJECTION must be 0 or 1"
#endif
#if GPU_POOL_STATIC_DIRECTORY != 0 && GPU_POOL_STATIC_DIRECTORY != 1
#error "GPU_POOL_STATIC_DIRECTORY must be 0 or 1"
#endif
#if GPU_POOL_CACHED_PROBABILITY != 0 && GPU_POOL_CACHED_PROBABILITY != 1
#error "GPU_POOL_CACHED_PROBABILITY must be 0 or 1"
#endif
#if GPU_DIRECTORY_HAS_BASE_ONLY != 0 && GPU_DIRECTORY_HAS_BASE_ONLY != 1
#error "GPU_DIRECTORY_HAS_BASE_ONLY must be 0 or 1"
#endif
#if GPU_DIRECTORY_OWNS_TILE_POLICY != 0 && GPU_DIRECTORY_OWNS_TILE_POLICY != 1
#error "GPU_DIRECTORY_OWNS_TILE_POLICY must be 0 or 1"
#endif
#if GPU_POOL_CACHED_PROBABILITY && !GPU_POOL_STATIC_DIRECTORY
#error "cached pool probability requires a static directory"
#endif
#if GPU_DIRECTORY_HAS_BASE_ONLY && !GPU_POOL_STATIC_DIRECTORY
#error "base-only directory requires static directory dispatch"
#endif
#if GPU_DIRECTORY_OWNS_TILE_POLICY && !GPU_DIRECTORY_HAS_BASE_ONLY
#error "directory-owned tile policy requires base-only directory support"
#endif
#if GPU_OPERATOR_THERMAL
#ifndef GPU_CONTACT_SEPARATE_AGE
#error "missing thermal entry policy GPU_CONTACT_SEPARATE_AGE"
#endif
#ifndef GPU_THERMAL_RELAX_NATIVE_ORDER
#error "missing thermal entry policy GPU_THERMAL_RELAX_NATIVE_ORDER"
#endif
#ifndef GPU_CONTACT_AGE
#error "missing thermal entry policy GPU_CONTACT_AGE"
#endif
#ifndef GPU_CONTACT_TIME_ZERO
#error "missing thermal entry policy GPU_CONTACT_TIME_ZERO"
#endif
#if GPU_CONTACT_SEPARATE_AGE != 0 && GPU_CONTACT_SEPARATE_AGE != 1
#error "GPU_CONTACT_SEPARATE_AGE must be 0 or 1"
#endif
#if GPU_THERMAL_RELAX_NATIVE_ORDER != 0 && GPU_THERMAL_RELAX_NATIVE_ORDER != 1
#error "GPU_THERMAL_RELAX_NATIVE_ORDER must be 0 or 1"
#endif
#endif
static_assert(std::is_same<GPU_OPERATOR_REAL, float>::value || std::is_same<GPU_OPERATOR_REAL, double>::value,
    "operator real must use binary32 or binary64 storage");
static_assert(std::is_same<GPU_OPERATOR_TIME, double>::value, "operator time must remain binary64 double");
#if !GPU_OPERATOR_THERMAL
static_assert(sizeof(GPU_OPERATOR_REAL) == 8, "nonthermal resident backend requires binary64");
#endif
// Function-like macros retain their documented lexical adapter role. They must be
// declared even when the selected backend intentionally supplies an empty action.
