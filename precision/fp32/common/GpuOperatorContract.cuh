#pragma once
#include <type_traits>

#ifndef GPU_OPERATOR_GEOMETRY
#define GPU_OPERATOR_GEOMETRY double
#endif
#ifndef GPU_OPERATOR_G
#define GPU_OPERATOR_G(x) static_cast<GPU_OPERATOR_GEOMETRY>(x)
#endif
